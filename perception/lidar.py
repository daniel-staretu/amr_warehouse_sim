"""
perception/lidar.py
-------------------
Simulated 3-D spinning LiDAR using PyBullet rayTestBatch.

Sensor model (VLP-16 inspired)
  Horizontal : 360 rays at 1 deg spacing  — full 360 deg sweep
  Vertical   : 16 channels, evenly spaced from LIDAR_VERT_MIN_DEG to
               LIDAR_VERT_MAX_DEG  (-15 to +15 deg)
  Total rays : 5 760 per scan
  Max range  : 20 m
  Rate       : LIDAR_INTERVAL sim steps  (10 Hz at 240 Hz)
  Mounting   : rigid cylinder visual on robot roof, sensor centre at
               LIDAR_SENSOR_HEIGHT above ground

Output per scan
  points  np.ndarray  (N, 4)  float32   x  y  z  intensity  — sensor frame
  labels  np.ndarray  (N,)    int32     per-point semantic class derived from
                                        the PyBullet body ID returned by each ray

Saved format
  Each scan is written as two NumPy .npy files when COLLECT_LIDAR_DATA is True:
    scans/{frame:06d}.npy   float32  (N, 4)
    labels/{frame:06d}.npy  int32    (N,)   0=background 1=shelf 2=crate 3=forklift
  An 80/20 train/val split is applied: every 5th frame goes to val/.

Sensor view (SENSOR_VIEW_ENABLED = True)
  Opens an OpenCV window with two panels:

  Top — Range image (720 x 192 px)
    360 horizontal columns x 16 vertical rows.
    Each pixel = one ray.  Highest elevation at top (after vertical flip).
    Colour  : blue (close) → yellow (far), black = no return.

  Bottom — Bird's Eye View (720 x 720 px, +-15 m)
    Point cloud projected onto the ground plane.
    Robot at centre, facing up.  Each hit = one pixel.
    Colour  : blue (close) → yellow (far).
    Distance rings at 5 m / 10 m / 15 m.
"""

import math
import os

import cv2
import numpy as np
import pybullet as p

from config import (
    LIDAR_HORIZ_RAYS, LIDAR_VERT_RAYS,
    LIDAR_VERT_MIN_DEG, LIDAR_VERT_MAX_DEG,
    LIDAR_MAX_RANGE, LIDAR_SENSOR_HEIGHT,
    COLLECT_LIDAR_DATA, LIDAR_DATA_DIR,
    SENSOR_VIEW_ENABLED,
    GUI_MODE,
    MAX_DETECTION_RANGE, RESOLUTION,
    LABEL_BACKGROUND, LABEL_SHELF, LABEL_CRATE, LABEL_FORKLIFT,
)

# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------
_RI_SCALE_W = 2    # range-image horizontal upscale factor  → 360*2 = 720 px
_RI_SCALE_H = 30   # range-image vertical upscale factor   → 16*30 = 480 px
_BEV_PX     = 540  # cell size for each quadrant (total window: 1082 × 1082 px)
_BEV_EXT    = 15.0 # world extent shown in BEV (metres from centre)
_WIN_NAME   = 'Sensor View'

# BGR colours for the semantic mask overlay (index = label id)
_MASK_COLORS = np.array([
    [ 15,  15,  15],   # 0 background
    [ 40, 180,  40],   # 1 shelf      (green)
    [ 40,  40, 200],   # 2 crate      (red)
    [ 20, 150, 240],   # 3 forklift   (orange)
], dtype=np.uint8)


class LidarSensor:
    """Simulated spinning LiDAR attached to the robot roof."""

    # Cosmetic cylinder visual on the robot roof (no collision, no mass)
    _VIS_RADIUS = 0.07
    _VIS_HEIGHT = 0.10

    def __init__(self, robot_id, world):
        self.robot_id      = robot_id
        self._shelf_ids    = set(world.shelf_ids)
        self._crate_ids    = set(world.crate_ids)
        self._forklift_ids = set(world.forklift_ids)

        self._body_to_label: dict = {
            **{bid: LABEL_SHELF    for bid in self._shelf_ids},
            **{bid: LABEL_CRATE    for bid in self._crate_ids},
            **{bid: LABEL_FORKLIFT for bid in self._forklift_ids},
        }

        # ------------------------------------------------------------------
        # Pre-compute sensor-local ray direction unit vectors.
        # Layout: vert channel changes slowly (outer), horiz ray changes fast
        # (inner).  Ray index i = vert_idx * H + horiz_idx.
        # Shape: (LIDAR_VERT_RAYS * LIDAR_HORIZ_RAYS, 3)
        # ------------------------------------------------------------------
        horiz = np.linspace(0.0, 2.0 * math.pi, LIDAR_HORIZ_RAYS, endpoint=False)
        vert  = np.radians(
            np.linspace(LIDAR_VERT_MIN_DEG, LIDAR_VERT_MAX_DEG, LIDAR_VERT_RAYS)
        )
        cos_v = np.cos(vert)
        sin_v = np.sin(vert)
        cos_h = np.cos(horiz)
        sin_h = np.sin(horiz)

        dx = np.outer(cos_v, cos_h).ravel()
        dy = np.outer(cos_v, sin_h).ravel()
        dz = np.repeat(sin_v, LIDAR_HORIZ_RAYS)

        self._ray_dirs = np.stack([dx, dy, dz], axis=1).astype(np.float32)
        self._n_rays   = len(self._ray_dirs)

        # Geometric visual
        self._vis_id     = self._create_visual()
        self._dbg_pts_id = None

        if SENSOR_VIEW_ENABLED:
            cv2.namedWindow(_WIN_NAME, cv2.WINDOW_NORMAL)

        # Data collection
        self._frame = 0
        if COLLECT_LIDAR_DATA:
            for split in ('train', 'val'):
                for sub in ('scans', 'labels'):
                    os.makedirs(
                        os.path.join(LIDAR_DATA_DIR, split, sub), exist_ok=True
                    )
            self._write_dataset_yaml()
            self._frame = self._find_next_frame()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_crate(self, body_id: int) -> None:
        """Register a crate spawned after sensor construction."""
        self._crate_ids.add(body_id)
        self._body_to_label[body_id] = LABEL_CRATE

    def scan(self, camera_frame=None):
        """Cast all rays and return the labelled point cloud.

        Parameters
        ----------
        camera_frame : (rgb, mask) tuple or None
            Forward-camera capture from ImageCollector.  When provided and
            SENSOR_VIEW_ENABLED is True, the image is shown in the sensor window.

        Returns
        -------
        points : (N, 4) float32  — x y z intensity in sensor frame
                                   intensity = 1 - (range / max_range)
        labels : (N,)   int32   — per-point semantic class
        """
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        _, _, yaw = p.getEulerFromQuaternion(orn)
        ox, oy, oz = pos[0], pos[1], LIDAR_SENSOR_HEIGHT

        self._update_visual(ox, oy, oz, orn)

        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        Rz = np.array(
            [[cos_y, -sin_y, 0.0],
             [sin_y,  cos_y, 0.0],
             [0.0,    0.0,   1.0]],
            dtype=np.float32,
        )
        dirs_world = self._ray_dirs @ Rz.T

        origin   = np.array([ox, oy, oz], dtype=np.float32)
        from_pts = np.broadcast_to(origin, (self._n_rays, 3))
        to_pts   = from_pts + dirs_world * LIDAR_MAX_RANGE

        results = p.rayTestBatch(from_pts.tolist(), to_pts.tolist(), numThreads=4)

        range_img = np.zeros((LIDAR_VERT_RAYS, LIDAR_HORIZ_RAYS), np.float32)
        pts_list  = []
        lbl_list  = []

        for i, (obj_id, _link, frac, hit_pos, _normal) in enumerate(results):
            if frac >= 1.0 or obj_id < 0 or obj_id == self.robot_id:
                continue

            v = i // LIDAR_HORIZ_RAYS
            h = i % LIDAR_HORIZ_RAYS
            range_img[v, h] = frac * LIDAR_MAX_RANGE

            dx = hit_pos[0] - ox
            dy = hit_pos[1] - oy
            dz = hit_pos[2] - oz
            lx =  cos_y * dx + sin_y * dy
            ly = -sin_y * dx + cos_y * dy

            pts_list.append((lx, ly, dz, 1.0 - frac))
            lbl_list.append(self._body_to_label.get(obj_id, LABEL_BACKGROUND))

        if not pts_list:
            points = np.zeros((0, 4), np.float32)
            labels = np.zeros(0, np.int32)
        else:
            points = np.array(pts_list, np.float32)
            labels = np.array(lbl_list, np.int32)

        if COLLECT_LIDAR_DATA:
            self._save_frame(points, labels)

        if GUI_MODE and not SENSOR_VIEW_ENABLED:
            self._draw_debug_points(points, ox, oy, oz, cos_y, sin_y)

        if SENSOR_VIEW_ENABLED:
            self._render_view(range_img, points, camera_frame)

        self._frame += 1
        return points, labels

    def obstacles_in_world(self, points, robot_state):
        """Convert above-floor, within-range LiDAR hits to world obstacle positions.

        Parameters
        ----------
        points     : (N, 4) float32  sensor-frame x y z intensity
        robot_state: (rx, ry, yaw)

        Returns
        -------
        list of (wx, wy) floats — snapped to the planner grid and deduplicated.
        """
        if len(points) == 0:
            return []

        lx = points[:, 0]
        ly = points[:, 1]
        lz = points[:, 2]

        horiz_range = np.hypot(lx, ly)
        mask = (
            (lz > 0.05)
            & (lz < 2.45)
            & (horiz_range < MAX_DETECTION_RANGE)
        )
        if not mask.any():
            return []

        lx_m, ly_m = lx[mask], ly[mask]

        rx, ry, yaw = robot_state
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        wx = rx + cos_y * lx_m - sin_y * ly_m
        wy = ry + sin_y * lx_m + cos_y * ly_m

        gx = (np.round(wx / RESOLUTION) * RESOLUTION).tolist()
        gy = (np.round(wy / RESOLUTION) * RESOLUTION).tolist()
        return list(set(zip(gx, gy)))

    def close(self):
        """Destroy the OpenCV window if the view was open."""
        if SENSOR_VIEW_ENABLED:
            cv2.destroyWindow(_WIN_NAME)

    # ------------------------------------------------------------------
    # PyBullet 3-D debug display
    # ------------------------------------------------------------------

    def _create_visual(self):
        vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self._VIS_RADIUS,
            length=self._VIS_HEIGHT,
            rgbaColor=[0.10, 0.10, 0.10, 1.0],
        )
        return p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis,
            basePosition=[0.0, 0.0, LIDAR_SENSOR_HEIGHT],
        )

    def _update_visual(self, x, y, z, quat):
        p.resetBasePositionAndOrientation(self._vis_id, [x, y, z], quat)

    def _draw_debug_points(self, points, ox, oy, oz, cos_y, sin_y):
        if self._dbg_pts_id is not None:
            p.removeUserDebugItem(self._dbg_pts_id)
            self._dbg_pts_id = None
        if len(points) == 0:
            return
        lx, ly, lz = points[:, 0], points[:, 1], points[:, 2]
        wx = cos_y * lx - sin_y * ly + ox
        wy = sin_y * lx + cos_y * ly + oy
        wz = lz + oz
        world_pos = np.stack([wx, wy, wz], axis=1)

        dist = np.hypot(lx, ly)
        t = np.clip(dist / LIDAR_MAX_RANGE, 0.0, 1.0).astype(np.float64)
        # blue→green→yellow: saturated hues fill the mid-range
        r = np.where(t <= 0.5, 0.0,          2.0 * (t - 0.5))
        g = np.where(t <= 0.5, 2.0 * t,      1.0)
        b = np.where(t <= 0.5, 1.0 - 2.0*t,  0.0)
        colors = np.stack([r, g, b], axis=1)

        self._dbg_pts_id = p.addUserDebugPoints(
            world_pos.tolist(), colors.tolist(), pointSize=2
        )

    # ------------------------------------------------------------------
    # Sensor view rendering  (2 × 2 grid)
    # ------------------------------------------------------------------
    #
    #   ┌─────────────┬─────────────┐
    #   │  BEV        │  CAMERA RGB │
    #   ├─────────────┼─────────────┤
    #   │  RANGE IMG  │  SEG MASK   │
    #   └─────────────┴─────────────┘
    #
    # Each cell is _BEV_PX × _BEV_PX.  Non-square content is letterboxed.
    # ------------------------------------------------------------------

    def _render_view(self, range_img, points, camera_frame=None):
        """Compose and display the 2 × 2 sensor grid."""
        sz = _BEV_PX

        top_left = self._render_bev(points)
        bot_left = self._fit_to_cell(self._render_range_image(range_img))

        if camera_frame is not None:
            rgb, mask = camera_frame
            top_right = self._fit_to_cell(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            msk_img   = _MASK_COLORS[np.clip(mask, 0, len(_MASK_COLORS) - 1)]
            bot_right = self._fit_to_cell(msk_img)
        else:
            top_right = bot_right = np.zeros((sz, sz, 3), np.uint8)

        cv2.putText(top_right, 'CAMERA  RGB',   (8, 18),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (200, 200, 200), 1)
        cv2.putText(bot_right, 'SEMANTIC MASK', (8, 18),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (200, 200, 200), 1)

        for i, (name, color) in enumerate(
                [('shelf',    (40, 180,  40)),
                 ('crate',    (40,  40, 200)),
                 ('forklift', (20, 150, 240))]):
            y = sz - 10 - i * 14
            cv2.rectangle(bot_right, (6, y - 9), (16, y + 1), color, -1)
            cv2.putText(bot_right, name, (20, y),
                        cv2.FONT_HERSHEY_PLAIN, 0.75, (180, 180, 180), 1)

        sep_v = np.zeros((sz, 2, 3), np.uint8)
        sep_h = np.zeros((2, sz * 2 + 2, 3), np.uint8)
        grid  = np.vstack([
            np.hstack([top_left, sep_v, top_right]),
            sep_h,
            np.hstack([bot_left, sep_v, bot_right]),
        ])

        cv2.imshow(_WIN_NAME, grid)
        key = cv2.waitKey(1) & 0xFF
        window_alive = cv2.getWindowProperty(_WIN_NAME, cv2.WND_PROP_VISIBLE) >= 1
        if not window_alive or key in (ord('q'), 27):
            raise KeyboardInterrupt

    def _fit_to_cell(self, img):
        """Scale img to fill _BEV_PX × _BEV_PX, centred on a black background."""
        h, w  = img.shape[:2]
        scale = min(_BEV_PX / w, _BEV_PX / h)
        nw, nh = int(w * scale), int(h * scale)
        interp  = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
        resized = cv2.resize(img, (nw, nh), interpolation=interp)
        cell    = np.zeros((_BEV_PX, _BEV_PX, 3), np.uint8)
        y0 = (_BEV_PX - nh) // 2
        x0 = (_BEV_PX - nw) // 2
        cell[y0:y0 + nh, x0:x0 + nw] = resized
        return cell

    # --- Range image ------------------------------------------------------

    def _render_range_image(self, range_img):
        """
        Render the raw 16 x 360 scan grid as a scaled-up image.

        Rows    = vertical channels; top = highest elevation (after flipud).
        Columns = horizontal azimuth, 0 deg (robot forward) at left.
        Colour  = blue (close) → yellow (far).  Black = no return.

        Output: (16*_RI_SCALE_H) x (360*_RI_SCALE_W) x 3  uint8
        """
        H, W = range_img.shape
        out_h = H * _RI_SCALE_H
        out_w = W * _RI_SCALE_W
        canvas = np.zeros((out_h, out_w, 3), np.uint8)

        hit = range_img > 0
        if hit.any():
            rows, cols = np.where(hit)
            t = np.clip(range_img[rows, cols] / LIDAR_MAX_RANGE, 0.0, 1.0)
            b = np.where(t <= 0.5, 1.0 - 2.0*t, 0.0)
            g = np.where(t <= 0.5, 2.0 * t,     1.0)
            r = np.where(t <= 0.5, 0.0,          2.0 * (t - 0.5))
            # flip rows so highest elevation is at top
            flipped_rows = (H - 1 - rows) * _RI_SCALE_H + _RI_SCALE_H // 2
            px_cols      = cols * _RI_SCALE_W  + _RI_SCALE_W  // 2
            canvas[flipped_rows, px_cols, 0] = (b * 255).astype(np.uint8)
            canvas[flipped_rows, px_cols, 1] = (g * 255).astype(np.uint8)
            canvas[flipped_rows, px_cols, 2] = (r * 255).astype(np.uint8)

        cv2.putText(canvas, 'RANGE IMAGE  (azimuth x elevation)',
                    (8, 14), cv2.FONT_HERSHEY_PLAIN, 0.9, (200, 200, 200), 1)
        cv2.putText(canvas, f'+{LIDAR_VERT_MAX_DEG:.0f}deg',
                    (4, 26), cv2.FONT_HERSHEY_PLAIN, 0.75, (120, 120, 120), 1)
        cv2.putText(canvas, f'{LIDAR_VERT_MIN_DEG:.0f}deg',
                    (4, canvas.shape[0] - 4),
                    cv2.FONT_HERSHEY_PLAIN, 0.75, (120, 120, 120), 1)

        return canvas

    # --- Bird's Eye View -------------------------------------------------

    def _render_bev(self, points):
        """
        Project the point cloud top-down onto a square canvas.
        Each hit is rendered as a single pixel, coloured by distance.

        Sensor frame convention:
          +x forward  →  up   in image
          +y left     →  left in image

        Output: _BEV_PX x _BEV_PX x 3  uint8
        """
        sz    = _BEV_PX
        ext   = _BEV_EXT
        scale = sz / (2.0 * ext)
        cx = cy = sz // 2

        canvas = np.full((sz, sz, 3), 15, np.uint8)

        for r_m in (5, 10, 15):
            r_px = int(r_m * scale)
            cv2.circle(canvas, (cx, cy), r_px, (45, 45, 45), 1, cv2.LINE_AA)
            lx_ring = cx + r_px + 4
            if lx_ring < sz - 30:
                cv2.putText(canvas, f'{r_m}m', (lx_ring, cy - 4),
                            cv2.FONT_HERSHEY_PLAIN, 0.75, (65, 65, 65), 1)

        cv2.line(canvas, (cx, 0),  (cx, sz),  (30, 30, 30), 1)
        cv2.line(canvas, (0, cy),  (sz, cy),  (30, 30, 30), 1)

        if len(points) > 0:
            lx_arr = points[:, 0]
            ly_arr = points[:, 1]

            px_arr = np.clip((cx - ly_arr * scale).astype(int), 0, sz - 1)
            py_arr = np.clip((cy - lx_arr * scale).astype(int), 0, sz - 1)

            dist = np.hypot(lx_arr, ly_arr)
            t = np.clip(dist / LIDAR_MAX_RANGE, 0.0, 1.0).astype(np.float32)
            b = np.where(t <= 0.5, 1.0 - 2.0*t, 0.0)
            g = np.where(t <= 0.5, 2.0 * t,     1.0)
            r = np.where(t <= 0.5, 0.0,          2.0 * (t - 0.5))
            canvas[py_arr, px_arr, 0] = (b * 255).astype(np.uint8)  # B
            canvas[py_arr, px_arr, 1] = (g * 255).astype(np.uint8)  # G
            canvas[py_arr, px_arr, 2] = (r * 255).astype(np.uint8)  # R

        cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), -1)
        cv2.arrowedLine(canvas, (cx, cy), (cx, cy - 22),
                        (255, 255, 255), 2, tipLength=0.45)

        cv2.putText(canvas, f'BIRD\'S EYE VIEW  (+-{ext:.0f} m)',
                    (8, 18), cv2.FONT_HERSHEY_PLAIN, 0.9, (200, 200, 200), 1)

        return canvas

    # ------------------------------------------------------------------
    # Data persistence
    # ------------------------------------------------------------------

    def _find_next_frame(self) -> int:
        max_idx = -1
        for split in ('train', 'val'):
            scan_dir = os.path.join(LIDAR_DATA_DIR, split, 'scans')
            if not os.path.isdir(scan_dir):
                continue
            for fname in os.listdir(scan_dir):
                if fname.endswith('.npy'):
                    try:
                        max_idx = max(max_idx, int(fname[:-4]))
                    except ValueError:
                        pass
        return max_idx + 1

    def _save_frame(self, points, labels):
        split = 'val' if self._frame % 5 == 0 else 'train'
        stem  = f'{self._frame:06d}'
        np.save(os.path.join(LIDAR_DATA_DIR, split, 'scans',  stem + '.npy'), points)
        np.save(os.path.join(LIDAR_DATA_DIR, split, 'labels', stem + '.npy'), labels)

    def _write_dataset_yaml(self):
        yaml_path = os.path.join(LIDAR_DATA_DIR, 'dataset.yaml')
        if os.path.exists(yaml_path):
            return
        content = (
            "# AMR Warehouse LiDAR dataset\n"
            f"data_root: {LIDAR_DATA_DIR}\n"
            "\n"
            "# Per-point semantic labels derived from PyBullet body IDs.\n"
            "classes:\n"
            "  - {label: 0, name: background}\n"
            "  - {label: 1, name: shelf}\n"
            "  - {label: 2, name: crate}\n"
            "  - {label: 3, name: forklift}\n"
            "\n"
            "splits:\n"
            "  train: train/\n"
            "  val:   val/\n"
            "\n"
            "# scans/{frame:06d}.npy   float32  (N, 4)  x y z intensity\n"
            "# labels/{frame:06d}.npy  int32    (N,)    semantic class per point\n"
            "#\n"
            "# Coordinate frame: sensor (robot-relative)\n"
            "#   +x forward   +y left   +z up\n"
            f"#\n"
            f"# Sensor: {LIDAR_HORIZ_RAYS}H x {LIDAR_VERT_RAYS}V  "
            f"({LIDAR_VERT_MIN_DEG} to {LIDAR_VERT_MAX_DEG} deg)  "
            f"max range {LIDAR_MAX_RANGE} m\n"
        )
        os.makedirs(LIDAR_DATA_DIR, exist_ok=True)
        with open(yaml_path, 'w') as f:
            f.write(content)
