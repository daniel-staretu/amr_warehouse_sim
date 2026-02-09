import pybullet as p
import time
import cv2
import numpy as np

from config import *
from simulation.world import WarehouseWorld
from simulation.robot import WarehouseRobot
from navigation.planner import AStarPlanner
from navigation.controller import PurePursuitController
from perception.vision import VisionSystem


def main():
    # 1. Setup Simulation
    if GUI_MODE:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    world = WarehouseWorld()
    obstacles = world.build_walls()

    # Add a target for the robot to find
    target_pos_real = [5, 5]
    world.add_target_crate(target_pos_real[0], target_pos_real[1])

    # 2. Spawn Robot
    start_pos = [-8, -8]
    robot = WarehouseRobot(start_pos=[start_pos[0], start_pos[1], 0.1])

    # 3. Setup Navigation
    planner = AStarPlanner(RESOLUTION, MAP_WIDTH, MAP_HEIGHT)
    planner.set_obstacles(obstacles)

    # Plan path to a location NEAR the target (we don't want to crash into it)
    goal_pos = [4, 4]
    print("Planning path...")
    path = planner.plan(start_pos, goal_pos)

    if not path:
        print("No path found!")
        return

    controller = PurePursuitController()
    controller.set_path(path)

    # 4. Setup Vision
    vision = VisionSystem(robot.id)

    # 5. Debug Visualization (Draw path lines)
    for i in range(len(path) - 1):
        p.addUserDebugLine([path[i][0], path[i][1], 0.1],
                           [path[i + 1][0], path[i + 1][1], 0.1],
                           [0, 0, 1], 2)

    # 6. Main Loop
    print("Starting simulation...")
    while True:
        # Get Sensor Data
        robot_state = robot.get_state()

        # Get Camera Feed
        img_rgb = vision.get_camera_image()

        # Run "AI" Detection
        detections, debug_img = vision.detect_target(img_rgb)

        # Logic: If we see the target, stop. Else, follow path.
        if detections:
            print("Target Detected! Stopping.")
            v, omega = 0, 0
            cv2.putText(debug_img, "TARGET FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            v, omega = controller.compute_control(robot_state)

        # Apply Control
        robot.apply_control(v, omega)

        # Step Simulation
        p.stepSimulation()

        # Visualization
        cv2.imshow("Robot Vision", debug_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(TIME_STEP)

    p.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()