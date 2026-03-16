# AMR Warehouse Simulation

A PyBullet-based simulation of an Autonomous Mobile Robot (AMR) navigating a warehouse environment. The robot plans kinodynamically feasible paths using Hybrid A\*, follows them with a combined heading and cross-track controller, detects obstacles via an onboard camera, and replans in real time without stopping.

## Features

- **Hybrid A\* planner** - kinodynamically feasible planning for a differential-drive robot; respects minimum turning radius, inflates obstacles by robot radius (C-space), and includes pivot-turn primitives to escape heading deadlocks
- **Holonomic fallback planners** - A\* and D\* Lite available for comparison via `config.py`
- **Path-tangent controller** - tracks planned arc segments using heading error and cross-track correction, with speed tapering near the goal
- **Dynamic obstacle avoidance** - onboard camera detects coloured obstacles via HSV masking (placeholder for a real ML detector); detected positions are back-projected to world coordinates via a pure-geometry localiser and added to the planner's C-space at runtime
- **Async replanning** - when the current path is blocked, a background thread computes a detour while the robot keeps moving; the new path is spliced in from the robot's current position to preserve momentum
- **Proximity-based re-trigger guard** - replanning only fires when a genuinely new obstacle appears (>1 m from any previously seen one), preventing repeated replanning as the robot approaches the same object
- **Autonomous loop** - robot continuously picks random reachable goals and replans on arrival
- **28 x 30 m warehouse** - 6 shelf units in 3 N-S columns with open driving aisles and a central cross-aisle

## Project Structure

```
amr_warehouse_sim/
├── main.py                     # Entry point - simulation loop
├── config.py                   # All tunable parameters
├── requirements.txt
├── navigation/
│   ├── planner.py              # HybridAStarPlanner, DStarLitePlanner, AStarPlanner
│   ├── controller.py           # Path-tangent tracking controller
│   └── replanner.py            # replan_to_goal, find_detour, pick_random_goal
├── simulation/
│   ├── world.py                # WarehouseWorld - walls and shelving layout
│   └── robot.py                # WarehouseRobot - differential drive body
└── perception/
    ├── vision.py               # VisionSystem - camera capture and target detection
    └── localizer.py            # ObstacleLocalizer - depth back-projection to world coords
```

## Requirements

- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `pybullet`, `numpy`, `opencv-python`, `matplotlib`, `torch`.

## Running

```bash
python main.py
```

## Configuration

All settings live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `PLANNER` | `"hybrid_astar"` | `"hybrid_astar"`, `"dstar_lite"`, or `"astar"` |
| `MAP_WIDTH` / `MAP_HEIGHT` | `28` / `30` m | World dimensions |
| `RESOLUTION` | `0.5` m | Metres per grid cell |
| `MAX_SPEED` | `1.5` m/s | Robot linear speed |
| `MAX_STEERING` | `3.0` rad/s | Maximum angular velocity |
| `LOOKAHEAD_DISTANCE` | `1.5` m | Waypoint advancement threshold |
| `K_HEADING` | `3.0` | Heading error gain (rad/s per rad) |
| `K_CTE` | `1.5` | Cross-track error gain (rad/s per m) |
| `MAX_DETECTION_RANGE` | `8.0` m | Ignore obstacle detections beyond this distance |
| `GUI_MODE` | `True` | Set `False` for headless runs |

## Architecture Notes

### Hybrid A\* Planner

The planner operates over a 3-D state space `(x, y, θ)` with heading discretised into 16 bins (~22.5° each). At each expansion it generates:

- **7 forward arc primitives** at `MAX_SPEED` with angular velocities spanning `[−MAX_STEERING, +MAX_STEERING]`, giving a minimum turning radius of `R_min = MAX_SPEED / MAX_STEERING`
- **2 pivot-turn primitives** that rotate one heading bin in place (no position change), with a cost penalty to ensure they are only used when all forward arcs are blocked

Obstacles are pre-inflated by `ROBOT_RADIUS` before search (C-space approach), so collision checking is a single set lookup per node. Dynamic obstacles are held in a separate inflated set that is rebuilt each perception frame without touching the static map.

### Dynamic Obstacle Pipeline

1. **Detect** - `VisionSystem` captures an RGB + depth frame every 8 simulation steps
2. **Localise** - `ObstacleLocalizer` back-projects each bounding box through the depth buffer using projective geometry to recover a world `(x, y)` footprint position
3. **Update map** - detected positions are inflated and added to the planner's dynamic obstacle set
4. **Replan (async)** - if the current path intersects any dynamic obstacle cell and the detected position is new (>1 m from all previously replanned obstacles), a background thread runs `replan_to_goal`; if the direct route is still blocked, `find_detour` tries perpendicular bypass waypoints at increasing lateral offsets
5. **Splice** - when the thread completes, the new path is applied from the waypoint closest to the robot's current position (`set_path_near`), so the controller continues without a heading discontinuity

### Controller

At each timestep the controller computes:

- **Heading error** `e_θ` - difference between the robot's yaw and the current segment's tangent direction
- **Cross-track error** `e_cte` - signed perpendicular distance from the robot to the segment (positive = robot is right of path)
- **Steering**: `ω = K_HEADING · e_θ + K_CTE · e_cte`, clamped to `±MAX_STEERING`
- **Speed**: full `MAX_SPEED` except within `LOOKAHEAD_DISTANCE` of the goal, where it tapers linearly to zero

### Warehouse Layout

```
+──────────── 28 m ─────────────+
│                               │  <- end aisle
│  [shelf]  [shelf]  [shelf]    │  north units (y = 2 to 12)
│                               │  <- cross-aisle
│  [shelf]  [shelf]  [shelf]    │  south units (y = -12 to -2)
│                               │  <- end aisle
+───────────────────────────────+
  x:  -8       0       8
```

Each shelf unit is 1.5 m wide x 10 m long x 2.5 m tall. Aisles between columns are ~6.5 m wide.
