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
"""

import math
import os

import numpy as np
import pybullet as p

from config import (
    LIDAR_HORIZ_RAYS, LIDAR_VERT_RAYS,
    LIDAR_VERT_MIN_DEG, LIDAR_VERT_MAX_DEG,
    LIDAR_MAX_RANGE, LIDAR_SENSOR_HEIGHT,
    COLLECT_LIDAR_DATA, LIDAR_DATA_DIR,
    LABEL_BACKGROUND, LABEL_SHELF, LABEL_CRATE, LABEL_FORKLIFT,
    GUI_MODE,
)


class LidarSensor:
    """Simulated spinning LiDAR attached to the robot roof."""

    # LiDAR head visual  (cosmetic — no collision, no mass)
    _VIS_RADIUS = 0.07   # m
    _VIS_HEIGHT = 0.10   # m

    # Per-label RGB colours for the PyBullet debug point cloud display
    _LABEL_RGB = {
        LABEL_BACKGROUND: [0.55, 0.55, 0.55],
        LABEL_SHELF:      [0.20, 0.45, 1.00],
        LABEL_CRATE:      [1.00, 0.20, 0.20],
        LABEL_FORKLIFT:   [0.89, 0.61, 0.14],
    }

    def __init__(self, robot_id, world):
        self.robot_id      = robot_id
        self._shelf_ids    = set(world.shelf_ids)
        self._crate_ids    = set(world.crate_ids)
        self._forklift_ids = set(world.forklift_ids)

        # ------------------------------------------------------------------
        # Pre-compute sensor-local ray direction unit vectors.
        # Layout: vert channel changes slowly (outer), horiz ray changes fast
        # (inner), matching the physical scan order of a spinning LiDAR.
        # Shape: (LIDAR_VERT_RAYS * LIDAR_HORIZ_RAYS, 3)
        # ------------------------------------------------------------------
        horiz = np.linspace(0.0, 2.0 * math.pi, LIDAR_HORIZ_RAYS, endpoint=False)
        vert  = np.radians(
            np.linspace(LIDAR_VERT_MIN_DEG, LIDAR_VERT_MAX_DEG, LIDAR_VERT_RAYS)
        )
        cos_v = np.cos(vert)    # (V,)
        sin_v = np.sin(vert)    # (V,)
        cos_h = np.cos(horiz)   # (H,)
        sin_h = np.sin(horiz)   # (H,)

        dx = np.outer(cos_v, cos_h).ravel()   # (V*H,)
        dy = np.outer(cos_v, sin_h).ravel()
        dz = np.repeat(sin_v, LIDAR_HORIZ_RAYS)

        self._ray_dirs = np.stack([dx, dy, dz], axis=1).astype(np.float32)
        self._n_rays   = len(self._ray_dirs)

        # ------------------------------------------------------------------
        # Geometric visual — cylinder snapped to the robot top each scan
        # ------------------------------------------------------------------
        self._vis_id     = self._create_visual()
        self._dbg_pts_id = None   # handle to the current debug point cloud

        # ------------------------------------------------------------------
        # Data collection bookkeeping
        # ------------------------------------------------------------------
        self._frame = 0
        if COLLECT_LIDAR_DATA:
            for split in ('train', 'val'):
                for sub in ('scans', 'labels'):
                    os.makedirs(
                        os.path.join(LIDAR_DATA_DIR, split, sub),
                        exist_ok=True,
                    )
            self._write_dataset_yaml()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self):
        """Cast all rays from the current sensor position and return the scan.

        Returns
        -------
        points : np.ndarray, shape (N, 4), float32
            Per-hit point in the *sensor frame*:
            columns  x (forward), y (left), z (up), intensity.
        labels : np.ndarray, shape (N,), int32
            Semantic class per point.
        """
        # ---- Sensor origin in world frame --------------------------------
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        _, _, yaw = p.getEulerFromQuaternion(orn)
        ox, oy, oz = pos[0], pos[1], LIDAR_SENSOR_HEIGHT

        # Move the cosmetic cylinder to follow the robot
        self._update_visual(ox, oy, oz, orn)

        # ---- Rotate pre-computed local ray directions by robot yaw -------
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        # Only the XY plane rotates; Z (up) is unchanged.
        Rz = np.array(
            [[cos_y, -sin_y, 0.0],
             [sin_y,  cos_y, 0.0],
             [0.0,    0.0,   1.0]],
            dtype=np.float32,
        )
        dirs_world = self._ray_dirs @ Rz.T   # (N, 3)

        # ---- Build ray endpoints -----------------------------------------
        origin  = np.array([ox, oy, oz], dtype=np.float32)
        from_pts = np.broadcast_to(origin, (self._n_rays, 3))
        to_pts   = from_pts + dirs_world * LIDAR_MAX_RANGE

        results = p.rayTestBatch(
            from_pts.tolist(), to_pts.tolist(), numThreads=4
        )

        # ---- Classify each hit -------------------------------------------
        pts_list    = []
        labels_list = []

        for i, (obj_id, _link, frac, hit_pos, _normal) in enumerate(results):
            if frac >= 1.0 or obj_id < 0:
                continue                     # ray missed everything
            if obj_id == self.robot_id:
                continue                     # self-intersection

            # Semantic label + simulated reflectivity intensity
            if obj_id in self._shelf_ids:
                label, intensity = LABEL_SHELF,      0.50
            elif obj_id in self._crate_ids:
                label, intensity = LABEL_CRATE,      0.80
            elif obj_id in self._forklift_ids:
                label, intensity = LABEL_FORKLIFT,   0.60
            else:
                label, intensity = LABEL_BACKGROUND, 0.30

            # World → sensor frame  (translate then un-rotate by yaw)
            dx = hit_pos[0] - ox
            dy = hit_pos[1] - oy
            dz = hit_pos[2] - oz
            sx_local =  cos_y * dx + sin_y * dy
            sy_local = -sin_y * dx + cos_y * dy

            pts_list.append((sx_local, sy_local, dz, intensity))
            labels_list.append(label)

        if not pts_list:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.int32)

        points = np.array(pts_list,    dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int32)

        # ---- Persist and visualise ---------------------------------------
        if COLLECT_LIDAR_DATA:
            self._save_frame(points, labels)

        if GUI_MODE:
            self._draw_points(points, labels, ox, oy, oz, cos_y, sin_y)

        self._frame += 1
        return points, labels

    # ------------------------------------------------------------------
    # Visual helpers
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

    # ------------------------------------------------------------------
    # Debug point cloud display
    # ------------------------------------------------------------------

    def _draw_points(self, points, labels, ox, oy, oz, cos_y, sin_y):
        """Replace the previous scan's debug points with the new ones."""
        if self._dbg_pts_id is not None:
            p.removeUserDebugItem(self._dbg_pts_id)
            self._dbg_pts_id = None

        if len(points) == 0:
            return

        # Sensor frame → world frame for display
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
    # Data persistence
    # ------------------------------------------------------------------

    def _save_frame(self, points, labels):
        """Write one scan to disk as a pair of .npy files.

        80 / 20 train / val split: every 5th frame (frame % 5 == 0) → val.
        """
        split = 'val' if self._frame % 5 == 0 else 'train'
        stem  = f'{self._frame:06d}'
        np.save(os.path.join(LIDAR_DATA_DIR, split, 'scans',  stem + '.npy'), points)
        np.save(os.path.join(LIDAR_DATA_DIR, split, 'labels', stem + '.npy'), labels)

    def _write_dataset_yaml(self):
        """Write a dataset manifest readable by PointPillars / mmdet3d pipelines."""
        yaml_path = os.path.join(LIDAR_DATA_DIR, 'dataset.yaml')
        if os.path.exists(yaml_path):
            return
        content = (
            "# AMR Warehouse LiDAR dataset\n"
            f"data_root: {LIDAR_DATA_DIR}\n"
            "\n"
            "# Per-point semantic classes\n"
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
            "# Array shapes\n"
            "#   scans/{frame:06d}.npy   float32  (N, 4)  columns: x y z intensity\n"
            "#   labels/{frame:06d}.npy  int32    (N,)    per-point class index\n"
            "#\n"
            "# Coordinate frame: sensor (robot-relative)\n"
            "#   +x  robot forward\n"
            "#   +y  robot left\n"
            "#   +z  up\n"
            f"#\n"
            f"# Sensor model: {LIDAR_HORIZ_RAYS}H x {LIDAR_VERT_RAYS}V  "
            f"({LIDAR_VERT_MIN_DEG} to {LIDAR_VERT_MAX_DEG} deg vertical)  "
            f"max range {LIDAR_MAX_RANGE} m\n"
        )
        os.makedirs(LIDAR_DATA_DIR, exist_ok=True)
        with open(yaml_path, 'w') as f:
            f.write(content)
