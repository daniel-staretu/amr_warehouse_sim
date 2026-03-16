"""
navigation/replanner.py
-----------------------
High-level replanning helpers that sit above the planner primitives.

These functions operate on planner instances and robot state tuples; they have
no dependency on PyBullet or the simulation loop and can be unit-tested in
isolation.
"""

import math
import random

from config import MAP_WIDTH, MAP_HEIGHT


# Goal-selection constants — kept here so replanner.py is self-contained.
MIN_GOAL_DISTANCE = 3.0   # new goal must be at least this far from the robot (m)
MAP_MARGIN        = 1.0   # keep goals away from outer walls (m)
GOAL_CLEARANCE    = 0.0   # C-space inflation already provides robot-radius margin


def to_xy(path):
    """Strip optional heading component — returns list of (x, y) tuples.
    Accepts both (x, y) and (x, y, theta) waypoint formats."""
    return [(wp[0], wp[1]) for wp in path]


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
    Find a two-leg detour path: robot -> bypass point -> goal.

    For each detected obstacle the function tries bypass waypoints offset
    perpendicular (left and right) to the robot-to-goal direction.  The first
    combination where both legs plan successfully is returned as a single
    concatenated path.

    offsets: lateral distances (m) from the obstacle to try, smallest first.
    """
    rx, ry, ryaw = robot_state[0], robot_state[1], robot_state[2]
    gx, gy = goal_pos[0], goal_pos[1]

    to_goal    = math.atan2(gy - ry, gx - rx)
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

        if not planner.is_free(gx, gy, clearance=GOAL_CLEARANCE):
            continue

        path = planner.plan(robot_pos, [gx, gy], start_heading=robot_heading)
        if path:
            return list(path[-1]), path

    print("Warning: could not find a reachable goal after max attempts.")
    return None, []
