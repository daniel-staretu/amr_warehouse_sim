import os
import pybullet as p
import pybullet_data
from config import *

_SHELF_URDF = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'assets', 'shelf', 'shelf_bay.urdf')
)


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

    def _load_shelf_run(self, x, y_center, length=10.0):
        """Load 3-D shelf model instances tiled along Y to fill a shelf unit,
        and mark the covered grid cells as obstacles for the planner."""
        BAY_WIDTH = 1.04        # metres — natural width of one OBJ bay
        SHELF_W   = 1.5         # obstacle footprint width (E-W), matches layout
        n_bays = round(length / BAY_WIDTH)

        # Tile bays symmetrically around y_center
        y0 = y_center - (n_bays * BAY_WIDTH) / 2 + BAY_WIDTH / 2
        for i in range(n_bays):
            p.loadURDF(_SHELF_URDF,
                       basePosition=[x, y0 + i * BAY_WIDTH, 0.0],
                       useFixedBase=1)

        # Mark obstacle cells with 0.25 m clearance buffer around the shelf footprint
        CLEARANCE = 0.25
        min_x = x - SHELF_W / 2 - CLEARANCE
        max_x = x + SHELF_W / 2 + CLEARANCE
        min_y = y_center - length / 2 - CLEARANCE
        max_y = y_center + length / 2 + CLEARANCE
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
        Warehouse layout (28 m wide x 30 m tall, origin at centre):

        Outer walls (grey) form the perimeter.

        Shelving arranged in 5 N-S columns:
            x in {-10, -5, 0, 5, 10}
        Each column has two shelf units split by a 4 m cross-aisle at y = 0:
            north unit centred at y =  7  (y =  2 -> 12)
            south unit centred at y = -7  (y = -12 -> -2)

        This creates:
            - 3.5 m E-W driving aisles between shelf columns
            - 4 m cross-aisle running the full width at y = 0
            - 3 m end aisles at the north and south walls
            - 2.75 m side aisles at the east and west walls
        """

        WALL_COLOR = [0.55, 0.55, 0.55, 1.0]
        WALL_H     = 2.5

        # --- Outer perimeter walls ---
        hw = MAP_WIDTH  / 2   # 14
        hh = MAP_HEIGHT / 2   # 15
        t  = 1.0              # wall thickness

        self._add_box( 0,        hh - t/2,  MAP_WIDTH,  t, WALL_H, WALL_COLOR)  # north
        self._add_box( 0,       -hh + t/2,  MAP_WIDTH,  t, WALL_H, WALL_COLOR)  # south
        self._add_box( hw - t/2, 0,          t, MAP_HEIGHT, WALL_H, WALL_COLOR)  # east
        self._add_box(-hw + t/2, 0,          t, MAP_HEIGHT, WALL_H, WALL_COLOR)  # west

        # --- Shelving ---
        # Each shelf unit: 1.5 m wide (E-W) x 10 m long (N-S)
        SHELF_L = 10.0

        SHELF_COLS  = [-8, 0, 8]
        SHELF_NORTH =  7   # centre y of north unit
        SHELF_SOUTH = -7   # centre y of south unit

        for sx in SHELF_COLS:
            self._load_shelf_run(sx, SHELF_NORTH, SHELF_L)
            self._load_shelf_run(sx, SHELF_SOUTH, SHELF_L)

        return self.obstacles

    def add_target_crate(self, x, y):
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.25],
                                  rgbaColor=[1, 0, 0, 1])
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.25])
        p.createMultiBody(baseVisualShapeIndex=vis, baseCollisionShapeIndex=col,
                          basePosition=[x, y, 0.25])
