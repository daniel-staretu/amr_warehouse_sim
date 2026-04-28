import pybullet as p
import time
import math
import random
import threading

from config import *
from simulation.world import WarehouseWorld
from simulation.robot import WarehouseRobot
from navigation.planner import AStarPlanner, DStarLitePlanner, HybridAStarPlanner, KinoDStarLitePlanner
from navigation.controller import PurePursuitController
from navigation.replanner import to_xy, pick_random_goal, replan_to_goal, find_detour
from perception.lidar import LidarSensor
from perception.image_collector import ImageCollector
from perception.localizer import has_new_obstacle

# ---------------------------------------------------------------------------
# Obstacle test case
#
# When OBSTACLE_TEST_MODE is True the simulation uses a fixed first goal and
# places a single red crate dead-centre on the known route to that goal, so
# the replanning pipeline is triggered deterministically on the very first run.
#
# Route analysis (robot start = (-12, -13), goal = (10, 12)):
#   west aisle north  →  cross-aisle east (y ≈ 0)  →  east aisle north
#   Crate at (0, 0) sits squarely in the cross-aisle segment.
#
# Set to False to revert to random goals and multiple scattered crates.
# ---------------------------------------------------------------------------
OBSTACLE_TEST_MODE = False

TEST_FIRST_GOAL      = [10.0, 12.0]   # fixed northeast goal
TEST_BLOCKING_CRATE  = (0.0,  0.0)    # guaranteed to be on the first path

# Used when OBSTACLE_TEST_MODE is False
SCATTERED_CRATES = [
    (-4.0,  6.0),   # north aisle between x=-8 and x=0 columns
    ( 4.0, -6.0),   # south aisle between x=0 and x=8 columns
]

def pick_random_start(obstacles, near=None, radius=6.0):
    """Return a random free grid position for the robot spawn point.

    If *near* is given, only cells within *radius* metres are considered.
    Falls back to an unconstrained pick if no cells satisfy the constraint.
    """
    obs_set = set(obstacles)
    free = []
    x = -(MAP_WIDTH / 2)
    while x <= MAP_WIDTH / 2:
        y = -(MAP_HEIGHT / 2)
        while y <= MAP_HEIGHT / 2:
            if (x, y) not in obs_set:
                if near is None or math.hypot(x - near[0], y - near[1]) <= radius:
                    free.append((x, y))
            y += RESOLUTION
        x += RESOLUTION
    if not free:
        return pick_random_start(obstacles)  # relax constraint and retry
    return random.choice(free)


def draw_path(path):
    """Draw path as blue debug lines. Returns list of debug item IDs."""
    line_ids = []
    for i in range(len(path) - 1):
        lid = p.addUserDebugLine(
            [path[i][0], path[i][1], 0.1],
            [path[i + 1][0], path[i + 1][1], 0.1],
            [0, 0, 1], 2
        )
        line_ids.append(lid)
    return line_ids


def clear_path(line_ids):
    """Remove all debug lines from a previous path."""
    for lid in line_ids:
        p.removeUserDebugItem(lid)


def main():
    # 1. Setup Simulation
    # SENSOR_VIEW_ENABLED replaces the 3-D PyBullet window with the OpenCV
    # sensor panels, so connect headless in that case.
    if GUI_MODE and not SENSOR_VIEW_ENABLED:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    world = WarehouseWorld()
    obstacles = world.build_walls()

    # Spawn crates — unknown to planner at startup; discovered by camera at runtime
    if OBSTACLE_TEST_MODE:
        world.add_target_crate(*TEST_BLOCKING_CRATE)
        print(f"[Test] Blocking crate at {TEST_BLOCKING_CRATE}  |  "
              f"fixed goal: {TEST_FIRST_GOAL}")
    else:
        for cx, cy in SCATTERED_CRATES:
            world.add_target_crate(cx, cy)

    # 2. Spawn Robot
    if OBSTACLE_TEST_MODE:
        start_xy = (-12.0, -13.0)
    elif COLLECT_LIDAR_DATA and world.forklift_ids:
        fpos, _ = p.getBasePositionAndOrientation(world.forklift_ids[0])
        start_xy = pick_random_start(obstacles, near=(fpos[0], fpos[1]), radius=6.0)
    else:
        start_xy = pick_random_start(obstacles)
    print(f"Start position: ({start_xy[0]:.2f}, {start_xy[1]:.2f})")
    robot           = WarehouseRobot(start_pos=[start_xy[0], start_xy[1], 0.1])
    lidar           = LidarSensor(robot.id, world)
    image_collector = ImageCollector(robot.id, world)

    # 3. Setup Navigation
    if PLANNER == "astar":
        planner = AStarPlanner(RESOLUTION, MAP_WIDTH, MAP_HEIGHT)
    elif PLANNER == "dstar_lite":
        planner = DStarLitePlanner(RESOLUTION, MAP_WIDTH, MAP_HEIGHT)
    elif PLANNER == "kino_dstar_lite":
        planner = KinoDStarLitePlanner(RESOLUTION, MAP_WIDTH, MAP_HEIGHT)
    else:
        planner = HybridAStarPlanner(RESOLUTION, MAP_WIDTH, MAP_HEIGHT)
    print(f"Planner: {PLANNER}")
    planner.set_obstacles(obstacles)

    # 4. Pick initial goal and plan path
    robot_state = robot.get_state()
    if OBSTACLE_TEST_MODE:
        path = planner.plan([robot_state[0], robot_state[1]],
                            TEST_FIRST_GOAL, start_heading=robot_state[2])
        goal_pos = TEST_FIRST_GOAL
        if not path:
            print("[Test] Warning: could not plan initial path to test goal.")
    else:
        goal_pos, path = pick_random_goal(planner,
                                          [robot_state[0], robot_state[1]],
                                          robot_state[2])
        while goal_pos is None:
            robot_state = robot.get_state()
            goal_pos, path = pick_random_goal(planner,
                                              [robot_state[0], robot_state[1]],
                                              robot_state[2])
    print(f"Initial goal: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")

    controller = PurePursuitController()
    controller.set_path(to_xy(path))
    line_ids = draw_path(to_xy(path))

    # 5. Main Loop
    print("Starting simulation... (Ctrl+C to quit)")
    step = 0
    last_replan_obs_pos = []
    _goal_retry_step = 0   # earliest step at which we may retry pick_random_goal

    # Async replan state — robot keeps moving while the planner runs in a thread
    _replan      = {'active': False, 'result': None, 'obstacles': []}
    _replan_lock = threading.Lock()

    def _replan_worker(rs, gp, obs_positions):
        new_path = replan_to_goal(planner, rs, gp)
        if not new_path:
            new_path = find_detour(planner, rs, gp, obs_positions)
        with _replan_lock:
            _replan['result'] = new_path or []
            _replan['active'] = False

    try:
        while True:
            robot_state = robot.get_state()

            # Apply a completed async replan
            with _replan_lock:
                replan_ready = _replan['result'] is not None
                replan_path  = _replan['result']
                replan_obs   = _replan['obstacles']
                if replan_ready:
                    _replan['result'] = None
            if replan_ready:
                if replan_path:
                    path = replan_path
                    clear_path(line_ids)
                    line_ids = draw_path(to_xy(path))
                    controller.set_path_near(to_xy(path),
                                             (robot_state[0], robot_state[1]))
                    last_replan_obs_pos = replan_obs
                    print("[LiDAR] Async replan applied.")
                else:
                    print("[LiDAR] Async replan found no path — will retry.")

            # Check if current goal has been reached
            dist_to_goal = math.hypot(goal_pos[0] - robot_state[0], goal_pos[1] - robot_state[1])
            if dist_to_goal < GOAL_THRESHOLD and step >= _goal_retry_step:
                print(f"Goal ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}) reached! Picking new goal...")
                clear_path(line_ids)
                new_goal, new_path = pick_random_goal(planner,
                                                      [robot_state[0], robot_state[1]],
                                                      robot_state[2])
                if new_goal is not None:
                    goal_pos, path = new_goal, new_path
                    last_replan_obs_pos = []
                    _goal_retry_step = 0
                    print(f"New goal: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
                    controller.set_path(to_xy(path))
                    line_ids = draw_path(to_xy(path))
                else:
                    # Back off for ~2 LiDAR scans so dynamic obstacles refresh
                    # before retrying; prevents a tight busy-loop of failed attempts.
                    _goal_retry_step = step + LIDAR_INTERVAL * 2
                    line_ids = []

            # Compute and apply control
            v, omega = controller.compute_control(robot_state)
            robot.apply_control(v, omega)

            # LiDAR scan + image capture + dynamic obstacle registration (10 Hz)
            if step % LIDAR_INTERVAL == 0:
                cam_frame = image_collector.capture(robot_state)
                points, _ = lidar.scan(camera_frame=cam_frame)

                if COLLECT_LIDAR_DATA and lidar._frame >= DATA_COLLECTION_SCANS:
                    print(f"Collected {DATA_COLLECTION_SCANS} scans — stopping.")
                    break

                with _replan_lock:
                    replan_active = _replan['active']

                if not replan_active:
                    planner.remove_dynamic_obstacles()
                    obs_positions = lidar.obstacles_in_world(points, robot_state)
                    for wx, wy in obs_positions:
                        planner.add_dynamic_obstacle(wx, wy)

                    if (obs_positions
                            and has_new_obstacle(obs_positions, last_replan_obs_pos)
                            and planner.path_is_blocked(path)):
                        print(f"[LiDAR] Path blocked — async replan to "
                              f"({goal_pos[0]:.2f}, {goal_pos[1]:.2f})...")
                        with _replan_lock:
                            _replan['active']    = True
                            _replan['result']    = None
                            _replan['obstacles'] = obs_positions
                        threading.Thread(
                            target=_replan_worker,
                            args=(robot_state, goal_pos, obs_positions),
                            daemon=True,
                        ).start()

            # Follow robot with GUI camera
            if GUI_MODE and not SENSOR_VIEW_ENABLED:
                p.resetDebugVisualizerCamera(
                    cameraDistance=15,
                    cameraYaw=0,
                    cameraPitch=-60,
                    cameraTargetPosition=[robot_state[0], robot_state[1], 0]
                )

            # Step Simulation
            p.stepSimulation()
            time.sleep(TIME_STEP)
            step += 1

    except KeyboardInterrupt:
        print("Simulation stopped.")
    finally:
        lidar.close()
        p.disconnect()


if __name__ == "__main__":
    main()
