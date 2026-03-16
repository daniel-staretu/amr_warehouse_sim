# AMR Warehouse Simulation

A PyBullet-based simulation of an Autonomous Mobile Robot (AMR) navigating a warehouse environment. The robot plans kinodynamically feasible paths using Hybrid A*, follows them with a path-tangent controller, and detects coloured targets via an onboard camera.

## Features

- **Hybrid A\* planner** - kinodynamically feasible planning for a differential-drive robot; respects minimum turning radius, inflates obstacles by robot radius (C-space), and includes pivot-turn primitives to escape heading deadlocks
- **Holonomic fallback planners** - A\* and D\* Lite available for comparison via `config.py`
- **Path-tangent controller** - tracks the planned arc segments directly using heading error and cross-track correction, preventing the corner-cutting that occurs with lookahead-based pursuit
- **Vision system** - robot-mounted camera with HSV-based colour detection (placeholder for a real ML detector), displayed live at ~30 fps
- **Autonomous loop** - robot continuously picks random reachable goals (with obstacle clearance checks) and replans on arrival
- **50 × 40 m warehouse** - 10 shelf units arranged in 5 N-S columns with 7.5 m driving aisles and a 6 m cross-aisle

## Project Structure

```
amr_warehouse_sim/
├── main.py                     # Entry point - simulation loop
├── config.py                   # All tunable parameters
├── requirements.txt
├── navigation/
│   ├── planner.py              # HybridAStarPlanner, DStarLitePlanner, AStarPlanner
│   └── controller.py           # Path-tangent tracking controller
├── simulation/
│   ├── world.py                # WarehouseWorld - walls and shelving layout
│   └── robot.py                # WarehouseRobot - differential drive body
└── perception/
    └── vision.py               # VisionSystem - camera capture and target detection
```

## Requirements

- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `pybullet`, `numpy`, `opencv-python`, `matplotlib`.

## Running

```bash
python main.py
```

## Configuration

All settings live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `PLANNER` | `"hybrid_astar"` | `"hybrid_astar"`, `"dstar_lite"`, or `"astar"` |
| `MAP_WIDTH` / `MAP_HEIGHT` | `50` / `40` m | World dimensions |
| `RESOLUTION` | `1` m | Metres per grid cell |
| `MAX_SPEED` | `3.5` m/s | Robot linear speed |
| `MAX_STEERING` | `3.0` rad/s | Maximum angular velocity |
| `LOOKAHEAD_DISTANCE` | `1.5` m | Waypoint advancement threshold |
| `K_HEADING` | `3.0` | Heading error gain (rad/s per rad) |
| `K_CTE` | `1.5` | Cross-track error gain (rad/s per m) |
| `GUI_MODE` | `True` | Set `False` for headless runs |

## Architecture Notes

### Hybrid A\* Planner

The planner operates over a 3-D state space `(x, y, θ)` with heading discretised into 16 bins (~22.5° each). At each expansion it generates:

- **7 forward arc primitives** at `MAX_SPEED` with angular velocities spanning `[−MAX_STEERING, +MAX_STEERING]`, giving a minimum turning radius of `R_min = MAX_SPEED / MAX_STEERING`
- **2 pivot-turn primitives** that rotate one heading bin in place (no position change), with a cost penalty to ensure they are only used when all forward arcs are blocked

Obstacles are pre-inflated by `ROBOT_RADIUS` before search (C-space approach), so collision checking is a single set lookup per node.

### Controller

The path-tangent controller tracks each planned arc segment directly rather than steering toward a lookahead point. At each timestep it computes:

- **Heading error** `e_θ` — difference between the robot's yaw and the current segment's tangent direction
- **Cross-track error** `e_cte` — signed perpendicular distance from the robot to the segment (positive = robot is right of path)
- **Steering**: `ω = K_HEADING · e_θ + K_CTE · e_cte`, clamped to `±MAX_STEERING`

### Warehouse Layout

```
+──────────────────── 50 m ─────────────────────+
│                                               │  ← 7 m end aisle
│  [shelf] [shelf] [shelf] [shelf] [shelf]      │  north units (y = 3 → 13)
│                                               │  ← 6 m cross-aisle
│  [shelf] [shelf] [shelf] [shelf] [shelf]      │  south units (y = -13 → -3)
│                                               │  ← 7 m end aisle
+───────────────────────────────────────────────+
  x: -18    -9      0      9      18
```

Each shelf unit is 1.5 m wide × 10 m long × 2.5 m tall. Driving aisles between columns are 7.5 m wide.
