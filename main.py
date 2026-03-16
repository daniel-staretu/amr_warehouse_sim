import pybullet as p
import time
import math
import random
import cv2

from config import *
from simulation.world import WarehouseWorld
from simulation.robot import WarehouseRobot
from navigation.planner import AStarPlanner, DStarLitePlanner, HybridAStarPlanner
from navigation.controller import PurePursuitController
from perception.vision import VisionSystem

MIN_GOAL_DISTANCE = 3.0   # meters — new goal must be at least this far from the robot
MAP_MARGIN = 1.5          # meters — keep goals away from the outer walls


GOAL_CLEARANCE = 1.0  # extra obstacle buffer (m) applied when sampling goals

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

    # 2. Spawn Robot (fixed starting position)
    start_pos = [-22, -16]
    robot = WarehouseRobot(start_pos=[start_pos[0], start_pos[1], 0.1])
    vision = VisionSystem(robot.id)

    # 3. Setup Navigation
    if PLANNER == "astar":
        planner = AStarPlanner(RESOLUTION, MAP_WIDTH, MAP_HEIGHT)
    elif PLANNER == "dstar_lite":
        planner = DStarLitePlanner(RESOLUTION, MAP_WIDTH, MAP_HEIGHT)
    else:
        planner = HybridAStarPlanner(RESOLUTION, MAP_WIDTH, MAP_HEIGHT)
    print(f"Planner: {PLANNER}")
    planner.set_obstacles(obstacles)

    # 4. Pick initial random goal and plan path
    robot_state = robot.get_state()
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
    controller.set_path(path)
    line_ids = draw_path(path)

    # 5. Main Loop
    print("Starting simulation... (Ctrl+C to quit)")
    step = 0
    try:
        while True:
            robot_state = robot.get_state()

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
                    print(f"New goal: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
                    controller.set_path(path)
                    line_ids = draw_path(path)
                else:
                    line_ids = []  # retry next iteration

            # Compute and apply control
            v, omega = controller.compute_control(robot_state)
            robot.apply_control(v, omega)

            # Camera perception — run every 8 steps (~30 fps from 240 Hz sim)
            if step % 8 == 0:
                frame = vision.get_camera_image()
                detections, annotated = vision.detect_target(frame)
                display = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                cv2.imshow("Robot Camera", display)
                cv2.waitKey(1)

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
