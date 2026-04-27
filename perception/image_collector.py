"""
perception/image_collector.py
------------------------------
Captures semantically labelled RGB images synchronized with LiDAR scans.

A single forward-facing camera is co-located with the LiDAR sensor at
LIDAR_SENSOR_HEIGHT, pointing in the robot's heading direction.

PyBullet's built-in segmentation rendering supplies pixel-perfect ground-truth
labels at zero extra cost — no heuristic colour masking needed:
  background  → LABEL_BACKGROUND (0)
  shelf       → LABEL_SHELF      (1)
  crate       → LABEL_CRATE      (2)
  forklift    → LABEL_FORKLIFT   (3)

Saved format (when COLLECT_IMAGE_DATA = True)
  Per scan, one file pair under the appropriate split:
    images/{frame:06d}.png   uint8  H x W x 3  BGR
    masks/{frame:06d}.png    uint8  H x W       semantic label index

  80 / 20 train / val split — every 5th frame goes to val/, matching the
  LiDAR pipeline so frame indices are aligned across both modalities.
"""

import math
import os

import cv2
import numpy as np
import pybullet as p

from config import (
    LIDAR_SENSOR_HEIGHT,
    LABEL_BACKGROUND, LABEL_SHELF, LABEL_CRATE, LABEL_FORKLIFT,
    COLLECT_IMAGE_DATA, IMAGE_DATA_DIR,
    IMAGE_COLLECTOR_WIDTH, IMAGE_COLLECTOR_HEIGHT, IMAGE_COLLECTOR_FOV,
    CAMERA_NEAR, CAMERA_FAR,
)


class ImageCollector:
    """Captures RGB + semantic-segmentation images in sync with LiDAR scans."""

    def __init__(self, robot_id, world):
        self.robot_id = robot_id

        self._proj = p.computeProjectionMatrixFOV(
            IMAGE_COLLECTOR_FOV,
            IMAGE_COLLECTOR_WIDTH / IMAGE_COLLECTOR_HEIGHT,
            CAMERA_NEAR,
            CAMERA_FAR,
        )

        # body_id → semantic label
        self._label_map: dict = {}
        for bid in world.shelf_ids:
            self._label_map[bid] = LABEL_SHELF
        for bid in world.crate_ids:
            self._label_map[bid] = LABEL_CRATE
        for bid in world.forklift_ids:
            self._label_map[bid] = LABEL_FORKLIFT

        self._frame = 0

        if COLLECT_IMAGE_DATA:
            for split in ('train', 'val'):
                for sub in ('images', 'masks'):
                    os.makedirs(
                        os.path.join(IMAGE_DATA_DIR, split, sub),
                        exist_ok=True,
                    )
            self._write_dataset_yaml()
            self._frame = self._find_next_frame()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_crate(self, body_id: int) -> None:
        """Register a crate spawned after sensor construction."""
        self._label_map[body_id] = LABEL_CRATE

    def capture(self, robot_state) -> tuple:
        """
        Capture a forward-facing image synchronized with a LiDAR scan.

        Parameters
        ----------
        robot_state : (rx, ry, yaw)

        Returns
        -------
        rgb  : uint8  (H, W, 3)   RGB channel order
        mask : uint8  (H, W)      semantic label per pixel
        """
        rx, ry, yaw = robot_state
        rgb, mask = self._capture_one(rx, ry, yaw)

        if COLLECT_IMAGE_DATA:
            self._save_frame(rgb, mask)

        self._frame += 1
        return rgb, mask

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _capture_one(self, rx, ry, cam_yaw):
        cam_pos = [rx, ry, LIDAR_SENSOR_HEIGHT]
        fwd     = [math.cos(cam_yaw), math.sin(cam_yaw), 0.0]
        target  = [rx + fwd[0], ry + fwd[1], LIDAR_SENSOR_HEIGHT]

        view = p.computeViewMatrix(cam_pos, target, [0.0, 0.0, 1.0])

        W, H = IMAGE_COLLECTOR_WIDTH, IMAGE_COLLECTOR_HEIGHT
        _, _, rgb_raw, _, seg_raw = p.getCameraImage(
            W, H,
            view, self._proj,
            renderer=p.ER_TINY_RENDERER,
        )

        rgb  = np.array(rgb_raw, dtype=np.uint8).reshape(H, W, 4)[:, :, :3]
        seg  = np.array(seg_raw, dtype=np.int32).reshape(H, W)

        mask = np.zeros((H, W), dtype=np.uint8)
        for bid, label in self._label_map.items():
            mask[seg == bid] = label

        return rgb, mask

    def _find_next_frame(self) -> int:
        max_idx = -1
        for split in ('train', 'val'):
            img_dir = os.path.join(IMAGE_DATA_DIR, split, 'images')
            if not os.path.isdir(img_dir):
                continue
            for fname in os.listdir(img_dir):
                if fname.endswith('.png'):
                    try:
                        max_idx = max(max_idx, int(fname[:-4]))
                    except ValueError:
                        pass
        return max_idx + 1

    def _save_frame(self, rgb, mask) -> None:
        split = 'val' if self._frame % 5 == 0 else 'train'
        stem  = f'{self._frame:06d}'

        img_path  = os.path.join(IMAGE_DATA_DIR, split, 'images', stem + '.png')
        mask_path = os.path.join(IMAGE_DATA_DIR, split, 'masks',  stem + '.png')
        cv2.imwrite(img_path,  cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, mask)

    def _write_dataset_yaml(self) -> None:
        yaml_path = os.path.join(IMAGE_DATA_DIR, 'dataset.yaml')
        if os.path.exists(yaml_path):
            return
        content = (
            "# AMR Warehouse image dataset\n"
            f"data_root: {IMAGE_DATA_DIR}\n"
            "\n"
            "# Semantic segmentation masks — label indices match LABEL_* constants.\n"
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
            "# images/{frame:06d}.png   uint8  H x W x 3  BGR\n"
            "# masks/{frame:06d}.png    uint8  H x W      label index per pixel\n"
            "#\n"
            "# Single forward-facing camera, aligned with robot heading.\n"
            "# Frame indices are aligned with the LiDAR scans dataset.\n"
            f"# Resolution: {IMAGE_COLLECTOR_WIDTH} x {IMAGE_COLLECTOR_HEIGHT}  "
            f"FOV: {IMAGE_COLLECTOR_FOV}° vertical  (~106° horizontal at 4:3)\n"
        )
        os.makedirs(IMAGE_DATA_DIR, exist_ok=True)
        with open(yaml_path, 'w') as f:
            f.write(content)
