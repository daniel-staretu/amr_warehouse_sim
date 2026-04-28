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

# ---------------------------------------------------------------------------
# LiDAR sensor  (VLP-16 inspired)
# ---------------------------------------------------------------------------
# Geometry
#   Horizontal : 360 rays at 1 deg spacing — full 360 deg sweep
#   Vertical   : 16 channels from -15 deg (floor side) to +15 deg (ceiling side)
#   Total rays : 360 x 16 = 5 760 per scan
#   Max range  : 20 m  — covers the full 28 x 30 m warehouse at oblique angles
#   Rate       : every LIDAR_INTERVAL sim steps = 10 Hz at 240 Hz
#
# Mounting
#   Sensor centre is LIDAR_SENSOR_HEIGHT above ground.
#   Robot body top is at z = ROBOT_BASE_Z + 0.05 = 0.15 m.
#   Cylinder visual: radius 0.07 m, height 0.10 m, centre at 0.20 m.
LIDAR_HORIZ_RAYS    = 360     # rays per revolution (1 deg resolution)
LIDAR_VERT_RAYS     = 16      # vertical channels
LIDAR_VERT_MIN_DEG  = -15.0   # bottom channel elevation (deg)
LIDAR_VERT_MAX_DEG  =  15.0   # top channel elevation (deg)
LIDAR_MAX_RANGE     = 20.0    # metres
LIDAR_SENSOR_HEIGHT =  0.20   # sensor centre z above ground (m)
LIDAR_INTERVAL      = 24      # scan every N sim steps  (10 Hz at 240 Hz)

# Semantic class indices used in saved label arrays and at inference time
LABEL_BACKGROUND = 0   # walls, floor, ceiling, unknown
LABEL_SHELF      = 1
LABEL_CRATE      = 2
LABEL_FORKLIFT   = 3

# ---------------------------------------------------------------------------
# Training data collection
# ---------------------------------------------------------------------------
# Set COLLECT_LIDAR_DATA = True to write point clouds to disk.
# Each scan is saved as two .npy files:
#   scans/{frame:06d}.npy   float32  (N, 4)  columns: x  y  z  intensity
#   labels/{frame:06d}.npy  int32    (N,)    per-point semantic class
# 80 / 20 train / val split (every 5th frame goes to val).
COLLECT_LIDAR_DATA    = True
LIDAR_DATA_DIR        = 'training_data/lidar'
DATA_COLLECTION_SCANS = 1500   # stop simulation automatically after this many frames

# ---------------------------------------------------------------------------
# Image data collection
# ---------------------------------------------------------------------------
# Set COLLECT_IMAGE_DATA = True to capture labelled RGB images in sync with
# each LiDAR scan.  A single forward-facing camera is co-located with the
# LiDAR at LIDAR_SENSOR_HEIGHT.
#
# Per scan, one file pair is written:
#   images/{frame:06d}.png   uint8  H x W x 3   BGR
#   masks/{frame:06d}.png    uint8  H x W        semantic label index
#
# Label indices match the LABEL_* constants above.
# 80 / 20 train / val split (every 5th frame → val/).
COLLECT_IMAGE_DATA      = True
IMAGE_DATA_DIR          = 'training_data/images'
IMAGE_COLLECTOR_WIDTH   = 640
IMAGE_COLLECTOR_HEIGHT  = 480
IMAGE_COLLECTOR_FOV     = 90   # vertical FOV (deg); gives ~106° horizontal at 4:3

# ---------------------------------------------------------------------------
# Sensor rendering view
# ---------------------------------------------------------------------------
# Set True to open a live OpenCV window showing two panels:
#   Top   — Range image : 360 x 16 raw scan grid (azimuth x elevation),
#            each pixel coloured by semantic class and dimmed by distance.
#   Bottom — Bird's Eye View : point cloud projected top-down +-15 m,
#            coloured by class with distance rings and a legend.
SENSOR_VIEW_ENABLED = True