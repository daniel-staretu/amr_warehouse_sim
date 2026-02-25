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

        rx, ry, ryaw = robot_state

        # 1. Advance last_index past any waypoints the robot has already reached.
        #    This prevents steering toward a point that is now behind the robot.
        while self.last_index < len(self.path) - 1:
            dist = math.hypot(self.path[self.last_index][0] - rx,
                              self.path[self.last_index][1] - ry)
            if dist < LOOKAHEAD_DISTANCE:
                self.last_index += 1
            else:
                break

        # 2. Find the first path point beyond the lookahead distance.
        #    Fall back to the final waypoint if all remaining points are closer.
        target_point = self.path[-1]
        for i in range(self.last_index, len(self.path)):
            dist = math.hypot(self.path[i][0] - rx, self.path[i][1] - ry)
            if dist >= LOOKAHEAD_DISTANCE:
                target_point = self.path[i]
                break

        # 3. Calculate heading error (alpha)
        tx, ty = target_point

        angle_to_target = math.atan2(ty - ry, tx - rx)
        alpha = angle_to_target - ryaw

        # Normalize angle to [-pi, pi]
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

        # 4. Compute controls
        # For pure pursuit: curvature = 2*sin(alpha) / Lookahead

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