import numpy as np

# Planner selection:
#   "hybrid_astar"     — kinodynamic Hybrid A* (recommended)
#   "kino_dstar_lite"  — D* Lite + Theta* smoothing + turning-radius arcs
#                        (designed for dynamic obstacle replanning)
#   "dstar_lite"       — plain holonomic D* Lite
#   "astar"            — holonomic A* baseline
PLANNER = "hybrid_astar"

# Simulation settings
TIME_STEP = 1./240.
GRAVITY = -9.8
GUI_MODE = True  # Set to False for headless training

# Map settings
MAP_WIDTH = 28   # meters
MAP_HEIGHT = 30  # meters
RESOLUTION = 0.5 # meters per grid cell (lower = finer path, slower calc)

# Robot settings
ROBOT_RADIUS = 0.3
WHEEL_RADIUS = 0.1
AXLE_LENGTH = 0.5
MAX_SPEED = 1.5     # m/s
MAX_STEERING = 3.0  # rad/s

# Navigation
LOOKAHEAD_DISTANCE = 1.5  # waypoint advancement threshold
GOAL_THRESHOLD     = 0.4  # distance to consider goal reached
K_HEADING = 3.0           # angular gain for heading error   (rad/s per rad)
K_CTE     = 1.5           # angular gain for cross-track error (rad/s per m)

# Vision
CAMERA_WIDTH  = 320
CAMERA_HEIGHT = 240
CAMERA_FOV    = 90       # vertical field of view, degrees
CAMERA_NEAR   = 0.1      # projection near plane (m)
CAMERA_FAR    = 100.0    # projection far plane (m)
CAMERA_Z_OFFSET = 0.5    # camera mount height above robot base (m)
ROBOT_BASE_Z    = 0.1    # robot spawn z — camera ground height = BASE_Z + Z_OFFSET

MAX_DETECTION_RANGE = 8.0  # ignore obstacle detections beyond this distance (m)

TARGET_COLOR_LOWER = np.array([0, 100, 100]) # Red color mask in HSV
TARGET_COLOR_UPPER = np.array([10, 255, 255])