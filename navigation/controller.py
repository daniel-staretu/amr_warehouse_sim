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

    def compute_control(self, robot_state):
        # robot_state: [x, y, yaw]
        if not self.path or self.last_index >= len(self.path):
            return 0.0, 0.0  # Stop

        # 1. Find the target point on the path (Lookahead)
        target_idx = self.last_index
        for i in range(self.last_index, len(self.path)):
            dist = math.hypot(self.path[i][0] - robot_state[0],
                              self.path[i][1] - robot_state[1])
            if dist > LOOKAHEAD_DISTANCE:
                target_idx = i
                break

        self.last_index = target_idx
        target_point = self.path[target_idx]

        # 2. Calculate heading error (alpha)
        tx, ty = target_point
        rx, ry, ryaw = robot_state

        angle_to_target = math.atan2(ty - ry, tx - rx)
        alpha = angle_to_target - ryaw

        # Normalize angle to [-pi, pi]
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

        # 3. Compute controls (Simple P-controller logic for differential drive)
        # For pure pursuit: curvature = 2*sin(alpha) / Lookahead
        # We simplify to linear/angular velocities

        linear_vel = MAX_SPEED
        angular_vel = 2.0 * linear_vel * math.sin(alpha) / LOOKAHEAD_DISTANCE

        # Cap Steering
        angular_vel = np.clip(angular_vel, -MAX_STEERING, MAX_STEERING)

        # Slow down if approaching end
        dist_to_goal = math.hypot(self.path[-1][0] - rx, self.path[-1][1] - ry)
        if dist_to_goal < LOOKAHEAD_DISTANCE:
            linear_vel = MAX_SPEED * (dist_to_goal / LOOKAHEAD_DISTANCE)
            if dist_to_goal < GOAL_THRESHOLD:
                linear_vel = 0
                angular_vel = 0

        return linear_vel, angular_vel