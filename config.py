import numpy as np

# Simulation settings
TIME_STEP = 1./240.
GRAVITY = -9.8
GUI_MODE = True  # Set to False for headless training

# Map settings
MAP_WIDTH = 20   # meters
MAP_HEIGHT = 20  # meters
RESOLUTION = 1 # meters per grid cell (lower = finer path, slower calc)

# Robot settings
ROBOT_RADIUS = 0.3
WHEEL_RADIUS = 0.1
AXLE_LENGTH = 0.5
MAX_SPEED = 15      # m/s
MAX_STEERING = 10.0  # rad/s

# Navigation
LOOKAHEAD_DISTANCE = 0.8  # Pure Pursuit lookahead
GOAL_THRESHOLD = 0.4      # Distance to consider goal reached

# Vision
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FOV = 90
TARGET_COLOR_LOWER = np.array([0, 100, 100]) # Red color mask in HSV
TARGET_COLOR_UPPER = np.array([10, 255, 255])