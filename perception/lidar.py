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
  labels  np.ndarray  (N,)    int32     per-point semantic class:
            LABEL_BACKGROUND = 0  (walls, floor, ceiling)
            LABEL_SHELF      = 1
            LABEL_CRATE      = 2
            LABEL_FORKLIFT   = 3

Saved format
  Each scan is written as two NumPy .npy files when COLLECT_LIDAR_DATA is True:
    scans/{frame:06d}.npy   float32  (N, 4)
    labels/{frame:06d}.npy  int32    (N,)
  .npy is directly loadable by PointNet++, PointPillars, and Improved
  PointPillars via np.load() with no conversion step.
  An 80/20 train/val split is applied: every 5th frame goes to val/.

LiDAR view (LIDAR_VIEW_ENABLED = True)
  Opens an OpenCV window with two panels:

  Top — Range image (720 x 192 px)
    360 horizontal columns x 16 vertical rows.
    Each pixel = one ray.  Highest elevation at top (after vertical flip).
    Colour  : semantic class BGR tinted by (1 - range/max_range),
              black = no return.

  Bottom — Bird's Eye View (720 x 720 px, +-15 m)
    Point cloud projected onto the ground plane.
    Robot at centre, facing up.  Coloured by class.
    Distance rings at 5 m / 10 m / 15 m.
    Class legend in the top-left corner.
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
    LIDAR_VIEW_ENABLED,
    LABEL_BACKGROUND, LABEL_SHELF, LABEL_CRATE, LABEL_FORKLIFT,
    GUI_MODE,
    MAX_DETECTION_RANGE, RESOLUTION,
)

# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------
_RI_SCALE_W = 2    # range-image horizontal upscale factor  → 360*2 = 720 px
_RI_SCALE_H = 12   # range-image vertical upscale factor   → 16*12 = 192 px
_BEV_PX     = 720  # bird's-eye-view canvas size (square)
_BEV_EXT    = 15.0 # world extent shown in BEV (metres from centre)
_WIN_NAME   = 'LiDAR View'


class LidarSensor:
    """Simulated spinning LiDAR attached to the robot roof."""

    # Cosmetic cylinder visual on the robot roof (no collision, no mass)
    _VIS_RADIUS = 0.07
    _VIS_HEIGHT = 0.10

    # Per-label colours used in PyBullet debug display (RGB, 0-1)
    _LABEL_RGB = {
        LABEL_BACKGROUND: [0.55, 0.55, 0.55],
        LABEL_SHELF:      [0.20, 0.45, 1.00],
        LABEL_CRATE:      [1.00, 0.20, 0.20],
        LABEL_FORKLIFT:   [0.89, 0.61, 0.14],
    }

    # Same colours in OpenCV BGR (0-255 uint8 tuples).
    # Derived from _LABEL_RGB: BGR = reversed RGB channels * 255.
    _LABEL_BGR = {
        LABEL_BACKGROUND: (140, 140, 140),
        LABEL_SHELF:      (255, 115,  51),
        LABEL_CRATE:      ( 51,  51, 255),
        LABEL_FORKLIFT:   ( 36, 156, 227),
    }

    # Human-readable names for the legend
    _LABEL_NAME = {
        LABEL_BACKGROUND: 'Background',
        LABEL_SHELF:      'Shelf',
        LABEL_CRATE:      'Crate',
        LABEL_FORKLIFT:   'Forklift',
    }

    def __init__(self, robot_id, world):
        self.robot_id      = robot_id
        self._shelf_ids    = set(world.shelf_ids)
        self._crate_ids    = set(world.crate_ids)
        self._forklift_ids = set(world.forklift_ids)

        # ------------------------------------------------------------------
        # Pre-compute sensor-local ray direction unit vectors.
        # Layout: vert channel changes slowly (outer), horiz ray changes fast
        # (inner).  Ray index i = vert_idx * H + horiz_idx — used to recover
        # (v, h) grid position when building the range image.
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

        # Create the OpenCV window up-front so the OS registers its close button
        # before the first frame arrives.  Without this, imshow auto-creates the
        # window each call and the WND_PROP_VISIBLE check is unreliable.
        if LIDAR_VIEW_ENABLED:
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self):
        """Cast all rays and return the classified point cloud.

        Also updates:
          - the cosmetic LiDAR cylinder visual
          - the PyBullet debug point cloud (if GUI_MODE)
          - the OpenCV LiDAR View window (if LIDAR_VIEW_ENABLED)
          - saved .npy files on disk (if COLLECT_LIDAR_DATA)

        Returns
        -------
        points : (N, 4) float32  — x y z intensity in sensor frame
        labels : (N,)   int32   — semantic class per point
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

        # Allocate range / label grids for the rendering view
        range_img = np.zeros((LIDAR_VERT_RAYS, LIDAR_HORIZ_RAYS), np.float32)
        label_img = np.full((LIDAR_VERT_RAYS, LIDAR_HORIZ_RAYS), -1, np.int8)

        pts_list, lbl_list = [], []

        for i, (obj_id, _link, frac, hit_pos, _normal) in enumerate(results):
            if frac >= 1.0 or obj_id < 0 or obj_id == self.robot_id:
                continue

            lbl, intensity = self._classify(obj_id)

            # Populate range/label image at grid position (v, h)
            v = i // LIDAR_HORIZ_RAYS
            h = i % LIDAR_HORIZ_RAYS
            range_img[v, h] = frac * LIDAR_MAX_RANGE
            label_img[v, h] = lbl

            # World → sensor frame
            dx = hit_pos[0] - ox
            dy = hit_pos[1] - oy
            dz = hit_pos[2] - oz
            lx =  cos_y * dx + sin_y * dy
            ly = -sin_y * dx + cos_y * dy

            pts_list.append((lx, ly, dz, intensity))
            lbl_list.append(lbl)

        if not pts_list:
            points = np.zeros((0, 4), np.float32)
            labels = np.zeros(0, np.int32)
        else:
            points = np.array(pts_list, np.float32)
            labels = np.array(lbl_list, np.int32)

        if COLLECT_LIDAR_DATA:
            self._save_frame(points, labels)

        if GUI_MODE and not LIDAR_VIEW_ENABLED:
            self._draw_debug_points(points, labels, ox, oy, oz, cos_y, sin_y)

        if LIDAR_VIEW_ENABLED:
            self._render_view(range_img, label_img, points, labels)

        self._frame += 1
        return points, labels

    def obstacles_in_world(self, points, robot_state):
        """Convert above-floor, within-range LiDAR hits to world obstacle positions.

        Purely geometric — no semantic classification needed.  Any physical
        return that is above the floor and within MAX_DETECTION_RANGE is treated
        as a potential obstacle, regardless of what object caused the return.

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

        lx = points[:, 0]   # forward in sensor frame
        ly = points[:, 1]   # left
        lz = points[:, 2]   # up

        horiz_range = np.hypot(lx, ly)
        mask = (
            (lz > 0.05)                          # above ground (exclude floor returns)
            & (lz < 2.45)                         # below ceiling
            & (horiz_range < MAX_DETECTION_RANGE)
        )
        if not mask.any():
            return []

        lx_m, ly_m = lx[mask], ly[mask]

        rx, ry, yaw = robot_state
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # Sensor frame → world frame
        wx = rx + cos_y * lx_m - sin_y * ly_m
        wy = ry + sin_y * lx_m + cos_y * ly_m

        # Snap to planner grid resolution and deduplicate
        gx = (np.round(wx / RESOLUTION) * RESOLUTION).tolist()
        gy = (np.round(wy / RESOLUTION) * RESOLUTION).tolist()
        return list(set(zip(gx, gy)))

    def close(self):
        """Destroy the OpenCV window if the view was open."""
        if LIDAR_VIEW_ENABLED:
            cv2.destroyWindow(_WIN_NAME)

    # ------------------------------------------------------------------
    # Classification helper
    # ------------------------------------------------------------------

    def _classify(self, obj_id):
        """Return (label, intensity) for a hit body ID."""
        if obj_id in self._shelf_ids:
            return LABEL_SHELF,      0.50
        if obj_id in self._crate_ids:
            return LABEL_CRATE,      0.80
        if obj_id in self._forklift_ids:
            return LABEL_FORKLIFT,   0.60
        return LABEL_BACKGROUND, 0.30

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

    def _draw_debug_points(self, points, labels, ox, oy, oz, cos_y, sin_y):
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
        default_rgb = self._LABEL_RGB[LABEL_BACKGROUND]
        colors = np.array(
            [self._LABEL_RGB.get(int(lb), default_rgb) for lb in labels],
            dtype=np.float64,
        )
        self._dbg_pts_id = p.addUserDebugPoints(
            world_pos.tolist(), colors.tolist(), pointSize=2
        )

    # ------------------------------------------------------------------
    # LiDAR view rendering
    # ------------------------------------------------------------------

    def _render_view(self, range_img, label_img, points, labels):
        """Compose and display the two-panel LiDAR view."""
        ri_panel  = self._render_range_image(range_img, label_img)
        bev_panel = self._render_bev(points, labels)

        # Thin black separator between panels
        sep = np.zeros((4, _BEV_PX, 3), np.uint8)

        # Resize range image panel to match BEV width (it should already be 720)
        if ri_panel.shape[1] != _BEV_PX:
            ri_panel = cv2.resize(ri_panel, (_BEV_PX, ri_panel.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        view = np.vstack([ri_panel, sep, bev_panel])
        cv2.imshow(_WIN_NAME, view)

        key = cv2.waitKey(1) & 0xFF
        # Stop the simulation if the window is closed (X button), Q, or Escape.
        # getWindowProperty returns < 1 once the OS has destroyed the window.
        window_alive = cv2.getWindowProperty(_WIN_NAME, cv2.WND_PROP_VISIBLE) >= 1
        if not window_alive or key in (ord('q'), 27):
            raise KeyboardInterrupt

    # --- Range image ------------------------------------------------------

    def _render_range_image(self, range_img, label_img):
        """
        Render the raw 16 x 360 scan grid as a scaled-up image.

        Rows    = vertical channels; top = highest elevation (after flipud).
        Columns = horizontal azimuth, 0 deg (robot forward) at left.
        Colour  = semantic label BGR, dimmed by normalised range.
        Black   = no return.

        Output: (16*_RI_SCALE_H) x (360*_RI_SCALE_W) x 3  uint8
        """
        H, W = range_img.shape   # 16, 360
        canvas = np.zeros((H, W, 3), np.uint8)

        for lbl, bgr in self._LABEL_BGR.items():
            mask = label_img == lbl
            if not mask.any():
                continue
            # Brightness falls off linearly with range; minimum 0.15 so far
            # objects are still visible (not confused with misses).
            brightness = np.clip(
                1.0 - range_img[mask] / LIDAR_MAX_RANGE, 0.15, 1.0
            )
            canvas[mask] = (np.array(bgr, np.float32) * brightness[:, None]).astype(np.uint8)

        # Highest elevation channel at top
        canvas = np.flipud(canvas)

        # Scale up with nearest-neighbour — raw sensor data, no interpolation
        canvas = cv2.resize(
            canvas,
            (W * _RI_SCALE_W, H * _RI_SCALE_H),
            interpolation=cv2.INTER_NEAREST,
        )

        # Axis labels
        cv2.putText(canvas, 'RANGE IMAGE  (azimuth x elevation)',
                    (8, 14), cv2.FONT_HERSHEY_PLAIN, 0.9, (200, 200, 200), 1)
        cv2.putText(canvas, f'+{LIDAR_VERT_MAX_DEG:.0f}deg',
                    (4, 26), cv2.FONT_HERSHEY_PLAIN, 0.75, (120, 120, 120), 1)
        cv2.putText(canvas, f'{LIDAR_VERT_MIN_DEG:.0f}deg',
                    (4, canvas.shape[0] - 4),
                    cv2.FONT_HERSHEY_PLAIN, 0.75, (120, 120, 120), 1)

        return canvas

    # --- Bird's Eye View -------------------------------------------------

    def _render_bev(self, points, labels):
        """
        Project the point cloud top-down onto a square canvas.

        Sensor frame convention:
          +x forward  →  up   in image
          +y left     →  left in image

        Output: _BEV_PX x _BEV_PX x 3  uint8
        """
        sz    = _BEV_PX
        ext   = _BEV_EXT
        scale = sz / (2.0 * ext)   # pixels per metre
        cx = cy = sz // 2

        canvas = np.full((sz, sz, 3), 15, np.uint8)  # near-black background

        # Distance rings
        for r_m in (5, 10, 15):
            r_px = int(r_m * scale)
            cv2.circle(canvas, (cx, cy), r_px, (45, 45, 45), 1, cv2.LINE_AA)
            # Label at the right side of each ring
            lx_ring = cx + r_px + 4
            if lx_ring < sz - 30:
                cv2.putText(canvas, f'{r_m}m', (lx_ring, cy - 4),
                            cv2.FONT_HERSHEY_PLAIN, 0.75, (65, 65, 65), 1)

        # Cross-hair
        cv2.line(canvas, (cx, 0),  (cx, sz),  (30, 30, 30), 1)
        cv2.line(canvas, (0, cy),  (sz, cy),  (30, 30, 30), 1)

        # Points — numpy batch indexing (much faster than per-point cv2.circle)
        if len(points) > 0:
            lx_arr = points[:, 0]   # forward
            ly_arr = points[:, 1]   # left

            # Sensor frame → image pixel
            #   +x (forward) → smaller py  (up in image)
            #   +y (left)    → smaller px  (left in image)
            px_arr = np.clip((cx - ly_arr * scale).astype(int), 0, sz - 1)
            py_arr = np.clip((cy - lx_arr * scale).astype(int), 0, sz - 1)

            for lbl, bgr in self._LABEL_BGR.items():
                mask = labels == lbl
                if not mask.any():
                    continue
                py_m, px_m = py_arr[mask], px_arr[mask]
                canvas[py_m, px_m] = bgr
                # 2x2 block for labelled objects — more visible than single pixel
                if lbl != LABEL_BACKGROUND:
                    canvas[np.clip(py_m + 1, 0, sz - 1), px_m] = bgr
                    canvas[py_m, np.clip(px_m + 1, 0, sz - 1)] = bgr
                    canvas[
                        np.clip(py_m + 1, 0, sz - 1),
                        np.clip(px_m + 1, 0, sz - 1),
                    ] = bgr

        # Robot marker — filled circle + forward arrow
        cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), -1)
        cv2.arrowedLine(canvas, (cx, cy), (cx, cy - 22),
                        (255, 255, 255), 2, tipLength=0.45)

        # Title
        cv2.putText(canvas, f'BIRD\'S EYE VIEW  (+-{ext:.0f} m)',
                    (8, 18), cv2.FONT_HERSHEY_PLAIN, 0.9, (200, 200, 200), 1)

        # Legend (top-left, below title)
        legend_order = [LABEL_BACKGROUND, LABEL_SHELF, LABEL_CRATE, LABEL_FORKLIFT]
        for i, lbl in enumerate(legend_order):
            y = 32 + i * 18
            bgr = self._LABEL_BGR[lbl]
            cv2.rectangle(canvas, (8, y - 8), (20, y + 4), bgr, -1)
            cv2.putText(canvas, self._LABEL_NAME[lbl], (26, y),
                        cv2.FONT_HERSHEY_PLAIN, 0.85, (185, 185, 185), 1)

        return canvas

    # ------------------------------------------------------------------
    # Data persistence
    # ------------------------------------------------------------------

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
            "classes:\n"
            f"  - {{label: {LABEL_BACKGROUND}, name: background}}\n"
            f"  - {{label: {LABEL_SHELF},      name: shelf}}\n"
            f"  - {{label: {LABEL_CRATE},      name: crate}}\n"
            f"  - {{label: {LABEL_FORKLIFT},   name: forklift}}\n"
            "\n"
            "splits:\n"
            "  train: train/\n"
            "  val:   val/\n"
            "\n"
            "# scans/{frame:06d}.npy   float32  (N, 4)  x y z intensity\n"
            "# labels/{frame:06d}.npy  int32    (N,)    class index\n"
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
