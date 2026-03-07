"""
Visualise the neural heuristic as a cost map.

For a fixed goal cell the script evaluates the trained MLP at every free grid
cell and renders the predicted cost as a matplotlib heatmap, with obstacles
shown in black.

Usage:
    python show_cost_map.py
"""

import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pybullet as p

# ── project imports ───────────────────────────────────────────────────────────
from config import MAP_WIDTH, MAP_HEIGHT, RESOLUTION
from simulation.world import WarehouseWorld
from navigation.neural_heuristic import (
    build_heuristic, encode_features, predict, ADMISSIBILITY_FACTOR,
)
from navigation.planner import NeuralDStarLitePlanner

# ── helpers ───────────────────────────────────────────────────────────────────

def world_to_grid(wx, wy, width, height, res):
    gx = int((wx + (width * res) / 2) / res)
    gy = int((wy + (height * res) / 2) / res)
    return gx, gy


def grid_to_world(gx, gy, width, height, res):
    wx = gx * res - (width * res) / 2
    wy = gy * res - (height * res) / 2
    return wx, wy


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # --- build world (DIRECT mode, no GUI needed) ----------------------------
    p.connect(p.DIRECT)
    world = WarehouseWorld()
    obstacle_world = world.build_walls()

    # Convert to grid-cell obstacle set (same as planner.set_obstacles)
    width  = int(MAP_WIDTH  / RESOLUTION)
    height = int(MAP_HEIGHT / RESOLUTION)
    obstacles = set()
    for ox, oy in obstacle_world:
        gx, gy = world_to_grid(ox, oy, width, height, RESOLUTION)
        if 0 <= gx < width and 0 <= gy < height:
            obstacles.add((gx, gy))

    # --- load / train heuristic ----------------------------------------------
    nn = build_heuristic(obstacles, width, height)
    if nn is None:
        print("ERROR: heuristic could not be built (no training data).")
        p.disconnect()
        sys.exit(1)

    norm = math.sqrt(width ** 2 + height ** 2)

    # --- choose a goal cell (world coords → grid) ----------------------------
    # Default: roughly centre-right of the open area
    goal_world = (6.0, 6.0)
    goal = world_to_grid(*goal_world, width, height, RESOLUTION)
    # If it lands on an obstacle, nudge to nearest free cell
    if goal in obstacles:
        for r in range(1, 10):
            found = False
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    c = (goal[0] + dx, goal[1] + dy)
                    if c not in obstacles and 0 <= c[0] < width and 0 <= c[1] < height:
                        goal = c
                        found = True
                        break
                if found:
                    break
            if found:
                break

    print(f"Goal grid cell : {goal}  "
          f"(world ~= {grid_to_world(*goal, width, height, RESOLUTION)})")

    # --- evaluate NN at every cell -------------------------------------------
    cost_grid = np.full((height, width), np.nan)

    for gy in range(height):
        for gx in range(width):
            if (gx, gy) in obstacles:
                continue
            feats = encode_features((gx, gy), goal, obstacles, width, height)
            raw   = predict(nn, feats) * norm
            cost_grid[gy, gx] = max(0.0, raw * ADMISSIBILITY_FACTOR)

    p.disconnect()

    # --- plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    # Cost heatmap (free cells only)
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color='black')   # obstacles → black
    masked = np.ma.masked_invalid(cost_grid)

    im = ax.imshow(
        masked,
        origin='lower',
        cmap=cmap,
        interpolation='nearest',
    )
    plt.colorbar(im, ax=ax, label='Predicted cost (world units)')

    # Goal marker
    ax.scatter([goal[0]], [goal[1]], marker='*', s=300, c='lime',
               zorder=5, label='Goal')

    ax.set_title('Neural Heuristic — Cost Map\n(black = obstacle, colour = predicted cost to goal)')
    ax.set_xlabel('Grid x')
    ax.set_ylabel('Grid y')
    ax.legend(loc='upper left')
    plt.tight_layout()
    out = 'cost_map.png'
    plt.savefig(out, dpi=150)
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
