"""
perception/localizer.py
-----------------------
Converts 2-D pixel-space detections into 3-D world-coordinate obstacle
positions using PyBullet's depth buffer and known camera geometry.

No machine-learning dependency — pure projective geometry.

Camera model (matches VisionSystem exactly)
-------------------------------------------
  - Mounted CAMERA_Z_OFFSET metres above the robot base.
  - Looks forward along the robot's heading (yaw), horizontal, up = (0,0,1).
  - Vertical FOV = CAMERA_FOV degrees.
  - Projection near / far = CAMERA_NEAR / CAMERA_FAR.

Coordinate conventions
-----------------------
  World  : x East, y North, z up  (PyBullet default)
  Eye    : x right, y up, z toward viewer  (OpenGL — camera looks along -z)
  Image  : u = column (left→right), v = row (top→bottom), origin top-left
"""

import math
import numpy as np

from config import (
    CAMERA_FOV, CAMERA_WIDTH, CAMERA_HEIGHT,
    CAMERA_NEAR, CAMERA_FAR,
    CAMERA_Z_OFFSET, ROBOT_BASE_Z,
    MAX_DETECTION_RANGE,
)


class ObstacleLocalizer:
    """
    Back-projects bounding-box detections to world (x, y) obstacle positions.

    Typical usage
    -------------
        localizer = ObstacleLocalizer()

        # Inside the perception loop:
        rgb, depth = vision.get_camera_data()
        detections = detector.detect(rgb)          # list of dicts with 'bbox' key
        obstacles  = localizer.detections_to_world(detections, depth, robot_state)
        # obstacles : list of (wx, wy) — feed directly into planner.add_dynamic_obstacle
    """

    def __init__(self):
        self._fov_y    = math.radians(CAMERA_FOV)     # vertical FOV in radians
        self._tan_half = math.tan(self._fov_y / 2.0)  # tan(45°) = 1.0 for 90° FOV
        self._aspect   = CAMERA_WIDTH / CAMERA_HEIGHT
        self._W        = CAMERA_WIDTH
        self._H        = CAMERA_HEIGHT
        self._near     = CAMERA_NEAR
        self._far      = CAMERA_FAR
        # Camera ground height = robot base z + mount offset
        self._cam_h    = ROBOT_BASE_Z + CAMERA_Z_OFFSET

    # ------------------------------------------------------------------
    # Depth linearisation
    # ------------------------------------------------------------------

    def linearise_depth(self, d):
        """
        Convert a raw PyBullet depth-buffer value d ∈ [0, 1] to a positive
        eye-space distance Z (metres from the camera lens).

        PyBullet stores depth as the standard OpenGL non-linear buffer:
            d = (far·(z - near)) / (z·(far - near))
        Inverting gives:
            Z = far·near / (far - d·(far - near))
        """
        return (self._far * self._near /
                (self._far - d * (self._far - self._near)))

    # ------------------------------------------------------------------
    # Single-pixel unprojection
    # ------------------------------------------------------------------

    def unproject_pixel(self, u, v, depth_buf, robot_state):
        """
        Back-project image pixel (u, v) to a 3-D world position.

        Parameters
        ----------
        u, v        : float — pixel column / row, 0-indexed, origin top-left
        depth_buf   : numpy array (H × W) of raw PyBullet depth values [0, 1]
        robot_state : (rx, ry, yaw) from robot.get_state()

        Returns
        -------
        (wx, wy, wz) as floats, or None if the pixel hits the far plane
        (no real geometry detected).
        """
        d = float(depth_buf[int(v), int(u)])
        if d >= 1.0:
            return None   # far-plane — sky / no geometry

        Z = self.linearise_depth(d)

        rx, ry, yaw = robot_state[0], robot_state[1], robot_state[2]

        # NDC coordinates of the pixel centre
        ndc_x = (2.0 * (u + 0.5) / self._W) - 1.0   # [-1, 1] left → right
        ndc_y = 1.0 - (2.0 * (v + 0.5) / self._H)   # [-1, 1] bottom → top

        # Eye-space coordinates (camera looks along -z_eye)
        x_eye = ndc_x * Z * self._tan_half * self._aspect
        y_eye = ndc_y * Z * self._tan_half

        # Camera basis vectors in world space
        #   forward (robot heading) : (cos yaw, sin yaw, 0)
        #   right                   : (sin yaw, -cos yaw, 0)
        #   up                      : (0, 0, 1)
        #
        # world = cam_pos
        #       + x_eye * right
        #       + y_eye * up
        #       + Z     * forward    (camera -z_eye points forward in world)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        wx = rx + x_eye * sin_yaw  + Z * cos_yaw
        wy = ry + x_eye * -cos_yaw + Z * sin_yaw
        wz = self._cam_h + y_eye

        return wx, wy, wz

    # ------------------------------------------------------------------
    # Bounding-box → ground obstacle position
    # ------------------------------------------------------------------

    def bbox_to_world(self, bbox_xyxy, depth_buf, robot_state):
        """
        Estimate the ground-plane (wx, wy) centre of an obstacle from its
        axis-aligned bounding box and the depth buffer.

        Strategy: sample the bottom strip of the bbox (most likely to touch
        the floor) plus the bbox centre.  Return the candidate whose world z
        is closest to ground (z ≈ 0), so the result represents the obstacle's
        footprint rather than its top surface.

        Parameters
        ----------
        bbox_xyxy   : (x1, y1, x2, y2) in pixel coordinates
        depth_buf   : numpy array (H × W) of raw PyBullet depth values
        robot_state : (rx, ry, yaw)

        Returns
        -------
        (wx, wy) or None if no valid depth sample is found.
        """
        x1, y1, x2, y2 = bbox_xyxy

        # Clamp to image bounds
        x1 = max(0, min(int(round(x1)), self._W - 1))
        x2 = max(0, min(int(round(x2)), self._W - 1))
        y1 = max(0, min(int(round(y1)), self._H - 1))
        y2 = max(0, min(int(round(y2)), self._H - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        u_mid = (x1 + x2) / 2.0
        v_mid = (y1 + y2) / 2.0

        # Candidate pixels: bottom strip + bbox centre
        candidates = [(u_mid, float(y2))]            # bottom centre
        candidates += [(u_mid, float(r))
                       for r in range(max(y1, y2 - 4), y2 + 1)]   # bottom strip
        candidates.append((u_mid, v_mid))             # bbox centre (fallback)

        world_pts = []
        for u, v in candidates:
            pt = self.unproject_pixel(u, v, depth_buf, robot_state)
            if pt is not None:
                world_pts.append(pt)

        if not world_pts:
            return None

        # Pick the point closest to z = 0 (ground level)
        best = min(world_pts, key=lambda p: abs(p[2]))
        return best[0], best[1]

    # ------------------------------------------------------------------
    # Batch conversion — main API
    # ------------------------------------------------------------------

    def detections_to_world(self, detections, depth_buf, robot_state):
        """
        Convert a list of detection dicts to world-coordinate obstacle positions.

        Each dict must contain a 'bbox' key.  Two formats are accepted:
          - (x, y, w, h)   — pixel-space top-left corner + size (cv2 convention)
          - (x1, y1, x2, y2) — pixel-space corners

        Detections whose bounding box is smaller than 4 × 4 pixels or whose
        estimated distance exceeds MAX_DETECTION_RANGE are discarded.

        Parameters
        ----------
        detections  : list of dicts, each with 'bbox' key
        depth_buf   : numpy array (H × W) of raw PyBullet depth values
        robot_state : (rx, ry, yaw)

        Returns
        -------
        list of (wx, wy) tuples — one per accepted detection.
        """
        rx, ry = robot_state[0], robot_state[1]
        positions = []

        for det in detections:
            bbox = det.get('bbox')
            if bbox is None or len(bbox) != 4:
                continue

            bx, by, bw, bh = bbox

            # bbox is always (x, y, w, h) — cv2 / VisionSystem convention.
            # YOLO (x1,y1,x2,y2) output must be converted to (x,y,w,h) by the
            # caller before passing to this method.
            x1, y1, x2, y2 = bx, by, bx + bw, by + bh

            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue   # too small to localise reliably

            result = self.bbox_to_world((x1, y1, x2, y2), depth_buf, robot_state)
            if result is None:
                continue

            dist = math.hypot(result[0] - rx, result[1] - ry)
            if dist <= MAX_DETECTION_RANGE:
                positions.append(result)

        return positions

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def pixel_depth(self, u, v, depth_buf):
        """Return the linearised depth (metres) at pixel (u, v). Useful for
        debugging — print this to verify the depth buffer is sensible."""
        d = float(depth_buf[int(v), int(u)])
        if d >= 1.0:
            return float('inf')
        return self.linearise_depth(d)
