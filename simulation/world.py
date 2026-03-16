import pybullet as p
import pybullet_data
from config import *


class WarehouseWorld:
    def __init__(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, GRAVITY)
        self.plane = p.loadURDF("plane.urdf")
        self.obstacles = []  # (x, y) tuples in world coordinates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_box(self, x, y, w, h, height, color):
        """Spawn a box at world (x, y) with footprint w×h and mark grid cells."""
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[w / 2, h / 2, height / 2],
            rgbaColor=color,
        )
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[w / 2, h / 2, height / 2],
        )
        p.createMultiBody(
            baseVisualShapeIndex=vis,
            baseCollisionShapeIndex=col,
            basePosition=[x, y, height / 2],
        )

        # Mark all grid cells covered by this footprint
        min_x, max_x = x - w / 2, x + w / 2
        min_y, max_y = y - h / 2, y + h / 2
        cx = min_x
        while cx <= max_x:
            cy = min_y
            while cy <= max_y:
                self.obstacles.append((cx, cy))
                cy += RESOLUTION
            cx += RESOLUTION

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------

    def build_walls(self):
        """
        Warehouse layout (50 m wide × 40 m tall, origin at centre):

        Outer walls (grey) form the perimeter.

        Shelving (brown) is arranged in 5 N-S columns:
            x ∈ {-18, -9, 0, 9, 18}
        Each column has two shelf units split by a 6 m cross-aisle at y = 0:
            north unit centred at y =  8  (y = 3 → 13)
            south unit centred at y = -8  (y = -13 → -3)

        This creates:
            - 7.5 m E-W driving aisles between every shelf column
            - 6 m cross-aisle running the full width at y ≈ 0
            - 7 m end aisles at the north and south walls
        """

        WALL_COLOR  = [0.55, 0.55, 0.55, 1.0]
        SHELF_COLOR = [0.55, 0.35, 0.15, 1.0]  # brown

        WALL_H  = 2.5
        SHELF_H = 2.5

        # --- Outer perimeter walls ---
        hw = MAP_WIDTH  / 2   # 25
        hh = MAP_HEIGHT / 2   # 20
        t  = 1.0              # wall thickness

        self._add_box( 0,       hh - t/2,  MAP_WIDTH,  t, WALL_H, WALL_COLOR)  # north
        self._add_box( 0,      -hh + t/2,  MAP_WIDTH,  t, WALL_H, WALL_COLOR)  # south
        self._add_box( hw - t/2, 0,         t, MAP_HEIGHT, WALL_H, WALL_COLOR) # east
        self._add_box(-hw + t/2, 0,         t, MAP_HEIGHT, WALL_H, WALL_COLOR) # west

        # --- Shelving ---
        # Each shelf unit: 1.5 m wide (E-W) × 10 m long (N-S)
        SHELF_W = 1.5   # width along X
        SHELF_L = 10.0  # length along Y

        SHELF_COLS  = [-18, -9, 0, 9, 18]
        SHELF_NORTH =  8   # centre y of north unit
        SHELF_SOUTH = -8   # centre y of south unit

        for sx in SHELF_COLS:
            self._add_box(sx, SHELF_NORTH, SHELF_W, SHELF_L, SHELF_H, SHELF_COLOR)
            self._add_box(sx, SHELF_SOUTH, SHELF_W, SHELF_L, SHELF_H, SHELF_COLOR)

        return self.obstacles

    def add_target_crate(self, x, y):
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.25],
                                  rgbaColor=[1, 0, 0, 1])
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.25])
        p.createMultiBody(baseVisualShapeIndex=vis, baseCollisionShapeIndex=col,
                          basePosition=[x, y, 0.25])
