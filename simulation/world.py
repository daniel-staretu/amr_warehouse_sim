import math
import os
import pybullet as p
import pybullet_data
from config import *

_ASSETS       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
_SHELF_OBJ    = os.path.join(_ASSETS, 'shelf',    'eb_metal_shelf_01_ds.obj')
_FORKLIFT_OBJ = os.path.join(_ASSETS, 'forklift', 'Forklift.obj')


class WarehouseWorld:
    def __init__(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, GRAVITY)
        self.plane = p.loadURDF("plane.urdf")
        p.changeVisualShape(self.plane, -1, textureUniqueId=-1,
                            rgbaColor=[0.45, 0.45, 0.45, 1.0])
        self.obstacles     = []   # (x, y) tuples in world coordinates
        self.shelf_ids     = []   # PyBullet body IDs — all shelf bay instances
        self.crate_ids     = []   # PyBullet body IDs — all target crates
        self.forklift_ids  = []   # PyBullet body IDs — all forklift instances

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
        """Spawn shelf bays with exact-mesh concave collision.

        GEOM_FORCE_CONCAVE_TRIMESH builds a BVH from the OBJ triangles, so
        LiDAR rays penetrate the open bay face and return hits on uprights,
        individual shelf boards, and the back panel rather than on a box hull.
        Only valid for static bodies (baseMass=0).
        """
        BAY_WIDTH = 1.04        # metres — natural width of one OBJ bay
        SHELF_W   = 1.5         # obstacle footprint width (E-W), matches layout
        n_bays = round(length / BAY_WIDTH)

        # Mesh transform matching the URDF visual origin:
        #   rpy = π/2  0  π/2   → OBJ Y-up axes to PyBullet Z-up axes
        #   xyz = -0.75  0  0   → centres the 1.5 m depth on the link origin
        shelf_quat  = p.getQuaternionFromEuler([math.pi / 2, 0.0, math.pi / 2])
        shelf_scale = [0.01, 0.015575, 0.038462]
        shelf_frame = [-0.75, 0.0, 0.0]

        y0 = y_center - (n_bays * BAY_WIDTH) / 2 + BAY_WIDTH / 2
        for i in range(n_bays):
            col = p.createCollisionShape(
                p.GEOM_MESH,
                fileName=_SHELF_OBJ,
                meshScale=shelf_scale,
                flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                collisionFramePosition=shelf_frame,
                collisionFrameOrientation=shelf_quat,
            )
            vis = p.createVisualShape(
                p.GEOM_MESH,
                fileName=_SHELF_OBJ,
                meshScale=shelf_scale,
                visualFramePosition=shelf_frame,
                visualFrameOrientation=shelf_quat,
            )
            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[x, y0 + i * BAY_WIDTH, 0.0],
            )
            self.shelf_ids.append(body_id)
            p.changeVisualShape(body_id, -1, rgbaColor=[0.22, 0.22, 0.22, 1.0])

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

    def _load_forklift(self, x, y, yaw=0.0):
        """Load the forklift with exact-mesh concave collision.

        GEOM_FORCE_CONCAVE_TRIMESH lets LiDAR rays resolve the actual forklift
        silhouette — forks, mast, cab — instead of a bounding-box hull.
        Returns the PyBullet body ID.
        """
        # Mesh transform matching the URDF visual origin:
        #   rpy = π/2  0  0   → OBJ Y-up axes to PyBullet Z-up axes
        #   xyz = -0.048  0  0 → centres the vehicle length on the link origin
        fl_quat   = p.getQuaternionFromEuler([math.pi / 2, 0.0, 0.0])
        fl_scale  = [0.004907, 0.004907, 0.004907]
        fl_frame  = [-0.048, 0.0, 0.0]
        body_quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])

        col = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=_FORKLIFT_OBJ,
            meshScale=fl_scale,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            collisionFramePosition=fl_frame,
            collisionFrameOrientation=fl_quat,
        )
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=_FORKLIFT_OBJ,
            meshScale=fl_scale,
            visualFramePosition=fl_frame,
            visualFrameOrientation=fl_quat,
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, 0.0],
            baseOrientation=body_quat,
        )
        p.changeVisualShape(body_id, linkIndex=-1,
                            rgbaColor=[0.890, 0.608, 0.141, 1.0])  # #E39B24
        self.forklift_ids.append(body_id)

        # Inflate the footprint by ROBOT_RADIUS so the planner treats it as
        # an impenetrable region (same convention as _load_shelf_run).
        FORKLIFT_L = 3.46   # total length (m)
        FORKLIFT_W = 1.24   # total width  (m)
        hl = FORKLIFT_L / 2 + ROBOT_RADIUS   # half-length + margin
        hw = FORKLIFT_W / 2 + ROBOT_RADIUS   # half-width  + margin

        # World-frame AABB of the rotated footprint (conservative).
        cos_y = abs(math.cos(yaw))
        sin_y = abs(math.sin(yaw))
        dx = hl * cos_y + hw * sin_y
        dy = hl * sin_y + hw * cos_y

        cx = x - dx
        while cx <= x + dx + 1e-9:
            cy = y - dy
            while cy <= y + dy + 1e-9:
                self.obstacles.append((cx, cy))
                cy += RESOLUTION
            cx += RESOLUTION

        return body_id

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

        WALL_COLOR = [0.78, 0.78, 0.78, 1.0]
        WALL_H     = 5.0

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
            if sx == 8:
                # NE corner: remove the north shelf row and park a forklift.
                # Forks face east (+X) toward the wall — a natural rest position.
                self._load_forklift(sx, SHELF_NORTH, yaw=0.0)
            else:
                self._load_shelf_run(sx, SHELF_NORTH, SHELF_L)
            self._load_shelf_run(sx, SHELF_SOUTH, SHELF_L)

        # Visual-only roof — no collision shape so LiDAR rays are unaffected
        ROOF_T = 0.3
        roof_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[MAP_WIDTH / 2, MAP_HEIGHT / 2, ROOF_T / 2],
            rgbaColor=[0.70, 0.70, 0.70, 1.0],
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=roof_vis,
            basePosition=[0.0, 0.0, WALL_H + ROOF_T / 2],
        )

        return self.obstacles

    def add_target_crate(self, x, y):
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.25],
                                  rgbaColor=[1, 0, 0, 1])
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.25])
        body_id = p.createMultiBody(baseVisualShapeIndex=vis,
                                    baseCollisionShapeIndex=col,
                                    basePosition=[x, y, 0.25])
        self.crate_ids.append(body_id)
        return body_id
