import pybullet as p
import time
import math
import random
import threading
import cv2

from config import *
from simulation.world import WarehouseWorld
from simulation.robot import WarehouseRobot
from navigation.planner import AStarPlanner, DStarLitePlanner, HybridAStarPlanner, KinoDStarLitePlanner
from navigation.controller import PurePursuitController
from perception.vision import VisionSystem
from perception.localizer import ObstacleLocalizer

MIN_GOAL_DISTANCE  = 3.0   # meters — new goal must be at least this far from the robot
MAP_MARGIN         = 1.0   # meters — keep goals away from the outer walls
GOAL_CLEARANCE     = 0.0   # C-space inflation already provides robot-radius safety margin

DETECTION_INTERVAL = 8     # run perception every N sim steps (~30 fps at 240 Hz)

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

def replan_to_goal(planner, robot_state, goal_pos):
    """Re-plan to the current goal from the current robot position.
    Returns a new path, or [] if the goal is now unreachable."""
    return planner.plan(
        [robot_state[0], robot_state[1]],
        goal_pos,
        start_heading=robot_state[2],
    )


def find_detour(planner, robot_state, goal_pos, obstacle_positions,
                offsets=(1.0, 1.5, 2.0)):
    """
    Find a two-leg detour path: robot → bypass point → goal.

    For each detected obstacle the function tries bypass waypoints offset
    perpendicular (left and right) to the robot-to-goal direction.  The first
    combination where both legs plan successfully is returned as a single
    concatenated path.

    offsets: lateral distances (m) from the obstacle to try, smallest first.
    """
    rx, ry, ryaw = robot_state[0], robot_state[1], robot_state[2]
    gx, gy = goal_pos[0], goal_pos[1]

    to_goal   = math.atan2(gy - ry, gx - rx)
    perp_left  = to_goal + math.pi / 2
    perp_right = to_goal - math.pi / 2

    for ox, oy in obstacle_positions:
        for offset in offsets:
            for perp in (perp_left, perp_right):
                bx = ox + offset * math.cos(perp)
                by = oy + offset * math.sin(perp)

                if not planner.is_free(bx, by):
                    continue

                leg1 = planner.plan([rx, ry], [bx, by], start_heading=ryaw)
                if not leg1:
                    continue

                # Heading at bypass point: toward the goal
                bypass_heading = math.atan2(gy - by, gx - bx)
                leg2 = planner.plan([bx, by], goal_pos,
                                    start_heading=bypass_heading)
                if not leg2:
                    continue

                # Stitch legs — drop duplicate junction point
                return leg1 + leg2[1:]

    return []


def pick_random_goal(planner, robot_pos, robot_heading=0.0, max_attempts=200):
    """Pick a random reachable goal. Returns (goal_pos, path) or (None, [])."""
    for _ in range(max_attempts):
        gx = random.uniform(-MAP_WIDTH / 2 + MAP_MARGIN, MAP_WIDTH / 2 - MAP_MARGIN)
        gy = random.uniform(-MAP_HEIGHT / 2 + MAP_MARGIN, MAP_HEIGHT / 2 - MAP_MARGIN)

        if math.hypot(gx - robot_pos[0], gy - robot_pos[1]) < MIN_GOAL_DISTANCE:
            continue

        # Reject goals that are inside or too close to any obstacle
        if not planner.is_free(gx, gy, clearance=GOAL_CLEARANCE):
            continue

        path = planner.plan(robot_pos, [gx, gy], start_heading=robot_heading)
        if path:
            return list(path[-1]), path

    print("Warning: could not find a reachable goal after max attempts.")
    return None, []


def has_new_obstacle(current_positions, known_positions, tolerance=1.0):
    """Return True if any position in current_positions is further than tolerance
    from every position in known_positions — i.e. it is a genuinely new obstacle.
    Prevents re-replanning when the localiser shifts its estimate of the same
    physical obstacle as the robot approaches it."""
    for cx, cy in current_positions:
        if not any(math.hypot(cx - kx, cy - ky) < tolerance
                   for kx, ky in known_positions):
            return True
    return False


def to_xy(path):
    """Strip optional heading component — returns list of (x, y) tuples.
    Accepts both (x, y) and (x, y, theta) waypoint formats."""
    return [(wp[0], wp[1]) for wp in path]


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
    if GUI_MODE:
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

    # 2. Spawn Robot (fixed starting position)
    start_pos = [-12, -13]
    robot = WarehouseRobot(start_pos=[start_pos[0], start_pos[1], 0.1])
    vision    = VisionSystem(robot.id)
    localizer = ObstacleLocalizer()

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
    last_replan_obs_pos = []   # world (wx, wy) positions at last successful replan trigger

    # Async replan state — robot keeps moving while the planner runs in a thread
    _replan = {'active': False, 'result': None, 'obstacles': frozenset()}
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

            # Apply a completed async replan — splice from current robot position
            # so the controller continues in the direction the robot is already moving
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
                    # Start tracking from the closest waypoint so there is no
                    # backward steer when the path was computed a few frames ago
                    controller.set_path_near(to_xy(path),
                                             (robot_state[0], robot_state[1]))
                    last_replan_obs_pos = replan_obs   # list of (wx, wy)
                    print("[Perception] Async replan applied.")
                else:
                    print("[Perception] Async replan found no path — will retry.")

            # Check if current goal has been reached
            dist_to_goal = math.hypot(goal_pos[0] - robot_state[0], goal_pos[1] - robot_state[1])
            if dist_to_goal < GOAL_THRESHOLD:
                print(f"Goal ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}) reached! Picking new goal...")
                clear_path(line_ids)
                new_goal, new_path = pick_random_goal(planner,
                                                      [robot_state[0], robot_state[1]],
                                                      robot_state[2])
                if new_goal is not None:
                    goal_pos, path = new_goal, new_path
                    last_replan_obs_pos = []
                    print(f"New goal: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
                    controller.set_path(to_xy(path))
                    line_ids = draw_path(to_xy(path))
                else:
                    line_ids = []  # retry next iteration

            # Compute and apply control
            v, omega = controller.compute_control(robot_state)
            robot.apply_control(v, omega)

            # Camera perception + dynamic obstacle replanning
            if step % DETECTION_INTERVAL == 0:
                rgb, depth = vision.get_camera_data()
                detections, annotated = vision.detect_target(rgb)

                # Display annotated feed
                cv2.imshow("Robot Camera", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

                with _replan_lock:
                    replan_active = _replan['active']

                # Only update the obstacle map when no replan thread is reading it
                if not replan_active:
                    planner.remove_dynamic_obstacles()
                    obstacle_positions = localizer.detections_to_world(
                        detections, depth, robot_state)
                    for wx, wy in obstacle_positions:
                        planner.add_dynamic_obstacle(wx, wy)

                    # Trigger async replan only when genuinely new obstacles appear.
                    # A 1.0 m proximity tolerance prevents repeated replanning when the
                    # localiser shifts its position estimate as the robot approaches the
                    # same physical obstacle.
                    if (obstacle_positions
                            and has_new_obstacle(obstacle_positions, last_replan_obs_pos)
                            and planner.path_is_blocked(path)):
                        print(f"[Perception] Path blocked — async replan to "
                              f"({goal_pos[0]:.2f}, {goal_pos[1]:.2f})...")
                        with _replan_lock:
                            _replan['active']    = True
                            _replan['result']    = None
                            _replan['obstacles'] = list(obstacle_positions)
                        threading.Thread(
                            target=_replan_worker,
                            args=(robot_state, goal_pos, list(obstacle_positions)),
                            daemon=True,
                        ).start()

            # Follow robot with GUI camera
            if GUI_MODE:
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
        cv2.destroyAllWindows()
        p.disconnect()


if __name__ == "__main__":
    main()
