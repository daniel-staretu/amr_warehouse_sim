"""
Neural heuristic for D* Lite.

A lightweight pure-numpy MLP is trained (once) to predict path cost from a
(node, goal, local-obstacle-patch) feature vector.  Weights are cached next
to this file and keyed by a hash of the obstacle set, so retraining only
happens when the map layout changes.

Architecture
------------
Input  : 28 features  [dx, dy, dist (3), 5×5 obstacle patch (25)]
Hidden : 64 → 32 (ReLU)
Output : 1  (normalised cost-to-goal, linear)

Training
--------
A standalone D* Lite is run on N_TRAIN_EPISODES random (start, goal) pairs.
After each run the full g-map (true cost from every expanded node to the goal)
is recorded as training labels.  Labels are normalised by the grid diagonal.

Adam (lr=1e-3) minimises MSE over 200 epochs with mini-batches of 256.
"""

import heapq
import hashlib
import math
import os

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHTS_DIR          = os.path.dirname(__file__)
PATCH_RADIUS         = 2                               # 5×5 = 25 cells
PATCH_SIZE           = (2 * PATCH_RADIUS + 1) ** 2
INPUT_DIM            = 3 + PATCH_SIZE                  # 28
LAYER_DIMS           = [INPUT_DIM, 64, 32, 1]
N_TRAIN_EPISODES     = 150
TRAIN_EPOCHS         = 200
TRAIN_LR             = 1e-3
TRAIN_BATCH          = 256
ADMISSIBILITY_FACTOR = 0.9   # scale NN output down for near-admissibility

_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1),
          (-1, -1), (-1, 1), (1, -1), (1, 1)]


# ---------------------------------------------------------------------------
# MLP — PyTorch nn.Module
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Tiny fully-connected network: ReLU hidden layers, linear output."""

    def __init__(self, layer_dims):
        super().__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):          # x: tensor (N, features)
        return self.net(x)

    def train_model(self, X, y, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, batch_size=TRAIN_BATCH):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        opt     = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.train()
        for epoch in range(epochs):
            total = 0.0
            for Xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self(Xb), yb)
                loss.backward()
                opt.step()
                total += loss.item()
            if (epoch + 1) % 50 == 0:
                print(f"  [NeuralHeuristic] epoch {epoch+1}/{epochs}  MSE={total/len(loader):.4f}")

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"  [NeuralHeuristic] weights saved -> {os.path.basename(path)}")

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict(nn_model, features):
    """Numpy feature vector → float prediction."""
    nn_model.eval()
    with torch.no_grad():
        t = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        return float(nn_model(t).squeeze().item())


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def encode_features(node, goal, obstacles, width, height):
    """
    28-D feature vector for a (node → goal) pair:
      [dx, dy, dist]  +  flattened 5×5 obstacle patch around *node*
    dx/dy normalised by grid dimensions; dist is the resulting Euclidean norm.
    """
    dx   = (goal[0] - node[0]) / width
    dy   = (goal[1] - node[1]) / height
    dist = math.sqrt(dx * dx + dy * dy)

    patch = []
    for py in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
        for px in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
            nx, ny = node[0] + px, node[1] + py
            if 0 <= nx < width and 0 <= ny < height:
                patch.append(1.0 if (nx, ny) in obstacles else 0.0)
            else:
                patch.append(1.0)   # out-of-bounds treated as obstacle

    return np.array([dx, dy, dist] + patch, dtype=np.float32)


# ---------------------------------------------------------------------------
# Standalone D* Lite for training-data collection
# ---------------------------------------------------------------------------

def _dstar_g_map(start, goal, obstacles, width, height):
    """
    Run D* Lite and return the full g-map: {node: true_cost_to_goal}.
    Returns an empty dict when no path exists from start to goal.
    """
    INF = float('inf')

    def h(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def nbrs(s):
        for dx, dy in _MOVES:
            nb = (s[0] + dx, s[1] + dy)
            if 0 <= nb[0] < width and 0 <= nb[1] < height and nb not in obstacles:
                yield nb, math.sqrt(dx * dx + dy * dy)

    g_map   = {}
    rhs_map = {goal: 0.0}
    open_l  = []
    open_d  = {}

    def key(s):
        mv = min(g_map.get(s, INF), rhs_map.get(s, INF))
        return (mv + h(start, s), mv)

    def push(s, k):
        heapq.heappush(open_l, (k, s))
        open_d[s] = k

    def update(s):
        if s != goal:
            rhs_map[s] = min(
                (c + g_map.get(sp, INF) for sp, c in nbrs(s)),
                default=INF,
            )
        open_d.pop(s, None)
        if g_map.get(s, INF) != rhs_map.get(s, INF):
            push(s, key(s))

    push(goal, key(goal))

    while open_l:
        k_old, s = heapq.heappop(open_l)
        if open_d.get(s) != k_old:
            continue
        if k_old >= key(start) and rhs_map.get(start, INF) == g_map.get(start, INF):
            break
        k_new = key(s)
        if k_old < k_new:
            push(s, k_new)
        elif g_map.get(s, INF) > rhs_map.get(s, INF):
            g_map[s] = rhs_map[s]
            del open_d[s]
            for sp, _ in nbrs(s):
                update(sp)
        else:
            g_map[s] = INF
            update(s)
            for sp, _ in nbrs(s):
                update(sp)

    return g_map if g_map.get(start, INF) < INF else {}


# ---------------------------------------------------------------------------
# Training-data collection
# ---------------------------------------------------------------------------

def collect_training_data(obstacles, width, height, n_episodes=N_TRAIN_EPISODES):
    """
    Generate (feature, label) pairs by running D* Lite on random start/goal
    pairs.  Labels are normalised by the grid diagonal so they lie in [0, 1].
    """
    norm       = math.sqrt(width ** 2 + height ** 2)
    free_cells = [(x, y) for x in range(width) for y in range(height)
                  if (x, y) not in obstacles]

    if len(free_cells) < 2:
        return (np.empty((0, INPUT_DIM), dtype=np.float32),
                np.empty(0,             dtype=np.float32))

    X, y    = [], []
    success = 0
    rng     = np.random.default_rng(1)
    attempts = 0

    while success < n_episodes and attempts < n_episodes * 5:
        attempts += 1
        idx   = rng.choice(len(free_cells), size=2, replace=False)
        start = free_cells[int(idx[0])]
        goal  = free_cells[int(idx[1])]

        g_map = _dstar_g_map(start, goal, obstacles, width, height)
        if not g_map:
            continue

        for node, cost in g_map.items():
            if cost < float('inf'):
                X.append(encode_features(node, goal, obstacles, width, height))
                y.append(cost / norm)
        success += 1

    print(f"  [NeuralHeuristic] collected {len(X)} samples "
          f"from {success}/{n_episodes} episodes")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def weights_path_for(obstacles):
    """Weights filename unique to this obstacle configuration."""
    key = hashlib.md5(str(sorted(obstacles)).encode()).hexdigest()[:12]
    return os.path.join(WEIGHTS_DIR, f"heuristic_{key}.pt")


def build_heuristic(obstacles, width, height):
    """
    Return a trained MLP for the given map.
    Loads cached weights when available; trains and saves otherwise.
    Returns None if training data could not be generated.
    """
    path = weights_path_for(obstacles)
    nn   = MLP(LAYER_DIMS)

    if os.path.exists(path):
        print(f"[NeuralHeuristic] loading cached weights: {os.path.basename(path)}")
        nn.load(path)
        return nn

    print(f"[NeuralHeuristic] no cache — training on {N_TRAIN_EPISODES} episodes …")
    X, y = collect_training_data(obstacles, width, height)

    if len(X) == 0:
        print("[NeuralHeuristic] no training data — falling back to Euclidean heuristic")
        return None

    nn.train_model(X, y)
    nn.save(path)
    return nn
