import math
import numpy as np
from config import *


class PurePursuitController:
    def __init__(self):
        self.path = []
        self.last_index = 0

    def set_path(self, path):
        self.path = path
        self.last_index = 0

    def set_path_near(self, path, robot_pos):
        """Like set_path but starts tracking from the waypoint closest to robot_pos.
        Use when splicing in a replanned path while the robot is still moving, so the
        controller doesn't try to steer back to already-passed waypoints."""
        self.path = path
        if not path:
            self.last_index = 0
            return
        rx, ry = robot_pos
        best = min(range(len(path)),
                   key=lambda i: math.hypot(path[i][0] - rx, path[i][1] - ry))
        # Never skip past second-to-last so there is always a next waypoint to track
        self.last_index = min(best, max(0, len(path) - 2))

    def compute_control(self, robot_state):
        if not self.path or len(self.path) < 2:
            return 0.0, 0.0

        rx, ry, ryaw = robot_state

        # --- Advance past waypoints the robot has already passed ---
        while self.last_index < len(self.path) - 1:
            dist = math.hypot(self.path[self.last_index][0] - rx,
                              self.path[self.last_index][1] - ry)
            if dist < LOOKAHEAD_DISTANCE * 0.5:
                self.last_index += 1
            else:
                break

        # --- Steering ---
        if self.last_index >= len(self.path) - 1:
            # Final waypoint: steer directly toward it
            tx, ty = self.path[-1]
            e_heading = math.atan2(ty - ry, tx - rx) - ryaw
            e_heading = (e_heading + math.pi) % (2 * math.pi) - math.pi
            angular_vel = np.clip(K_HEADING * e_heading, -MAX_STEERING, MAX_STEERING)
        else:
            # Safety: skip zero-length segments
            while (self.last_index < len(self.path) - 2 and
                   math.hypot(self.path[self.last_index + 1][0] - self.path[self.last_index][0],
                               self.path[self.last_index + 1][1] - self.path[self.last_index][1]) < 1e-6):
                self.last_index += 1

            i = self.last_index
            px, py = self.path[i]
            qx, qy = self.path[i + 1]

            # Heading error: robot vs path-segment tangent
            path_heading = math.atan2(qy - py, qx - px)
            e_heading = (path_heading - ryaw + math.pi) % (2 * math.pi) - math.pi

            # Cross-track error: signed perpendicular distance to segment.
            # Positive when the robot is to the right of the path direction.
            seg_dx, seg_dy = qx - px, qy - py
            seg_len = math.hypot(seg_dx, seg_dy)
            if seg_len > 1e-9:
                rnx =  seg_dy / seg_len
                rny = -seg_dx / seg_len
                e_cte = (rx - px) * rnx + (ry - py) * rny
            else:
                e_cte = 0.0

            angular_vel = np.clip(
                K_HEADING * e_heading + K_CTE * e_cte,
                -MAX_STEERING, MAX_STEERING
            )

        # --- Forward speed (taper near goal) ---
        dist_to_goal = math.hypot(self.path[-1][0] - rx, self.path[-1][1] - ry)
        if dist_to_goal < LOOKAHEAD_DISTANCE:
            linear_vel = MAX_SPEED * (dist_to_goal / LOOKAHEAD_DISTANCE)
            if dist_to_goal < GOAL_THRESHOLD:
                return 0.0, 0.0
        else:
            linear_vel = MAX_SPEED

        return linear_vel, angular_vel
