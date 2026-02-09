import pybullet as p
import pybullet_data
import math
from config import *


class WarehouseWorld:
    def __init__(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, GRAVITY)
        self.plane = p.loadURDF("plane.urdf")
        self.obstacles = []  # Stores (x, y) tuples in WORLD coordinates

    def build_walls(self):
        # Create Walls (Simple Boxes)
        wall_height = 1.0

        # Function to add a wall and mark grid cells
        def add_wall(x, y, w, h):
            # 1. Visual/Physics Body
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[w / 2, h / 2, wall_height / 2],
                                      rgbaColor=[0.8, 0.8, 0.8, 1])
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[w / 2, h / 2, wall_height / 2])
            p.createMultiBody(baseVisualShapeIndex=vis, baseCollisionShapeIndex=col,
                              basePosition=[x, y, wall_height / 2])

            # 2. Mark Grid Cells as Obstacles
            # We iterate through the wall's area using the grid resolution
            local_obs = []

            # Calculate bounds
            min_x = x - (w / 2)
            max_x = x + (w / 2)
            min_y = y - (h / 2)
            max_y = y + (h / 2)

            # Scan the area with a small buffer to ensure we catch edges
            curr_x = min_x
            while curr_x <= max_x:
                curr_y = min_y
                while curr_y <= max_y:
                    local_obs.append((curr_x, curr_y))
                    curr_y += RESOLUTION
                curr_x += RESOLUTION

            return local_obs

        # Outer Boundaries
        self.obstacles.extend(add_wall(0, 10, 20, 1))  # Top
        self.obstacles.extend(add_wall(0, -10, 20, 1))  # Bottom
        self.obstacles.extend(add_wall(10, 0, 1, 20))  # Right
        self.obstacles.extend(add_wall(-10, 0, 1, 20))  # Left

        # Internal Shelves/Obstacles
        self.obstacles.extend(add_wall(0, 0, 10, 2))  # Center Block

        return self.obstacles

    def add_target_crate(self, x, y):
        # A Red Box to be detected by Vision
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.25], rgbaColor=[1, 0, 0, 1])  # Red
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.25])
        p.createMultiBody(baseVisualShapeIndex=vis, baseCollisionShapeIndex=col, basePosition=[x, y, 0.25])