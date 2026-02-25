import heapq
import math


# ---------------------------------------------------------------------------
# D* Lite Planner
# ---------------------------------------------------------------------------

class DStarLitePlanner:
    """
    D* Lite (Koenig & Likhachev, 2002).

    Plans from goal → start so that replanning after obstacle changes is cheap.
    The public interface matches AStarPlanner: set_obstacles() + plan().
    """

    INF = float('inf')
    _MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def __init__(self, resolution, map_width, map_height):
        self.resolution = resolution
        self.width = int(map_width / resolution)
        self.height = int(map_height / resolution)
        self.obstacles = set()

        # per-plan state (reset in plan())
        self._g = {}
        self._rhs = {}
        self._open_list = []   # heap of (key, node)
        self._open_dict = {}   # node -> key currently in heap
        self._km = 0
        self._start = None
        self._goal = None

    # ------------------------------------------------------------------
    # Coordinate helpers (identical to AStarPlanner)
    # ------------------------------------------------------------------

    def world_to_grid(self, x, y):
        gx = int((x + (self.width * self.resolution) / 2) / self.resolution)
        gy = int((y + (self.height * self.resolution) / 2) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        wx = (gx * self.resolution) - (self.width * self.resolution) / 2
        wy = (gy * self.resolution) - (self.height * self.resolution) / 2
        return wx, wy

    def set_obstacles(self, obstacle_list):
        self.obstacles.clear()
        for ox, oy in obstacle_list:
            self.obstacles.add(self.world_to_grid(ox, oy))

    # ------------------------------------------------------------------
    # D* Lite internals
    # ------------------------------------------------------------------

    def _h(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _g(self, s):
        return self._g_map.get(s, self.INF)

    def _rhs(self, s):
        return self._rhs_map.get(s, self.INF)

    def _key(self, s):
        min_val = min(self._g_map.get(s, self.INF), self._rhs_map.get(s, self.INF))
        return (min_val + self._h(self._start, s) + self._km, min_val)

    def _neighbors(self, s):
        """Return list of (neighbor, edge_cost) for 8-connected grid."""
        result = []
        for dx, dy in self._MOVES:
            n = (s[0] + dx, s[1] + dy)
            if 0 <= n[0] < self.width and 0 <= n[1] < self.height and n not in self.obstacles:
                result.append((n, math.sqrt(dx * dx + dy * dy)))
        return result

    def _push(self, s, k):
        heapq.heappush(self._open_list, (k, s))
        self._open_dict[s] = k

    def _update_vertex(self, s):
        if s != self._goal:
            best = self.INF
            for sp, cost in self._neighbors(s):
                val = cost + self._g_map.get(sp, self.INF)
                if val < best:
                    best = val
            self._rhs_map[s] = best

        # Remove stale open-list entry
        self._open_dict.pop(s, None)

        if self._g_map.get(s, self.INF) != self._rhs_map.get(s, self.INF):
            self._push(s, self._key(s))

    def _compute_shortest_path(self):
        while self._open_list:
            k_old, s = heapq.heappop(self._open_list)

            # Skip stale heap entries
            if self._open_dict.get(s) != k_old:
                continue

            k_new = self._key(s)
            start_key = self._key(self._start)

            # Stop when start is locally consistent and nothing better remains
            if k_old >= start_key and self._rhs_map.get(self._start, self.INF) == self._g_map.get(self._start, self.INF):
                break

            if k_old < k_new:
                # Key has increased — reinsert with corrected key
                self._push(s, k_new)
            elif self._g_map.get(s, self.INF) > self._rhs_map.get(s, self.INF):
                # Overconsistent: lower g to rhs
                self._g_map[s] = self._rhs_map[s]
                del self._open_dict[s]
                for sp, _ in self._neighbors(s):
                    self._update_vertex(sp)
            else:
                # Underconsistent: raise g to INF and re-propagate
                self._g_map[s] = self.INF
                self._update_vertex(s)
                for sp, _ in self._neighbors(s):
                    self._update_vertex(sp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, start_pos, goal_pos):
        sx, sy = self.world_to_grid(start_pos[0], start_pos[1])
        gx, gy = self.world_to_grid(goal_pos[0], goal_pos[1])

        self.obstacles.discard((sx, sy))
        self.obstacles.discard((gx, gy))

        self._start = (sx, sy)
        self._goal = (gx, gy)
        self._km = 0
        self._g_map = {}
        self._rhs_map = {self._goal: 0.0}
        self._open_list = []
        self._open_dict = {}

        self._push(self._goal, self._key(self._goal))
        self._compute_shortest_path()

        if self._g_map.get(self._start, self.INF) == self.INF:
            return []  # No path found

        # Greedy path extraction: follow minimum-cost neighbours from start → goal
        path = []
        current = self._start
        visited = set()
        while current != self._goal:
            if current in visited:
                return []  # Cycle — should not happen with a consistent solution
            visited.add(current)
            path.append(self.grid_to_world(*current))

            best_node, best_cost = None, self.INF
            for sp, cost in self._neighbors(current):
                val = cost + self._g_map.get(sp, self.INF)
                if val < best_cost:
                    best_cost, best_node = val, sp
            if best_node is None:
                return []
            current = best_node

        path.append(self.grid_to_world(*self._goal))
        return path


# ---------------------------------------------------------------------------
# A* Planner (original)
# ---------------------------------------------------------------------------

class Node:
    def __init__(self, x, y, cost=0, heuristic=0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.heuristic = heuristic
        self.parent = parent

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)


class AStarPlanner:
    def __init__(self, resolution, map_width, map_height):
        self.resolution = resolution
        self.width = int(map_width / resolution)
        self.height = int(map_height / resolution)
        self.obstacles = set()

    def set_obstacles(self, obstacle_list):
        """ obstacle_list: list of (x, y) tuples in world coordinates """
        self.obstacles.clear()
        for (ox, oy) in obstacle_list:
            gx, gy = self.world_to_grid(ox, oy)
            self.obstacles.add((gx, gy))

    def world_to_grid(self, x, y):
        # Offset coordinates so (0,0) is at the center of the grid map
        gx = int((x + (self.width * self.resolution) / 2) / self.resolution)
        gy = int((y + (self.height * self.resolution) / 2) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        wx = (gx * self.resolution) - (self.width * self.resolution) / 2
        wy = (gy * self.resolution) - (self.height * self.resolution) / 2
        return wx, wy

    def plan(self, start_pos, goal_pos):
        sx, sy = self.world_to_grid(start_pos[0], start_pos[1])
        gx, gy = self.world_to_grid(goal_pos[0], goal_pos[1])

        # Ensure start and goal are not blocked by the grid's own resolution
        if (sx, sy) in self.obstacles: self.obstacles.remove((sx, sy))
        if (gx, gy) in self.obstacles: self.obstacles.remove((gx, gy))

        start_node = Node(sx, sy, 0, self.heuristic(sx, sy, gx, gy))
        open_list = []
        heapq.heappush(open_list, start_node)
        visited = set()
        visited.add((sx, sy))

        while open_list:
            current = heapq.heappop(open_list)

            if abs(current.x - gx) <= 1 and abs(current.y - gy) <= 1:
                return self.reconstruct_path(current)

            # 8-connectivity (Allowing diagonals)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = current.x + dx, current.y + dy

                # Movement cost: 1 for straight, 1.414 for diagonal
                move_cost = math.sqrt(dx ** 2 + dy ** 2)

                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) not in self.obstacles and (nx, ny) not in visited:
                        new_cost = current.cost + move_cost
                        new_node = Node(nx, ny, new_cost, self.heuristic(nx, ny, gx, gy), current)
                        heapq.heappush(open_list, new_node)
                        visited.add((nx, ny))

        return []

    def heuristic(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def reconstruct_path(self, node):
        path = []
        while node:
            path.append(self.grid_to_world(node.x, node.y))
            node = node.parent
        return path[::-1]  # Return reversed path (start -> end)


# ---------------------------------------------------------------------------
# Neural D* Lite Planner
# ---------------------------------------------------------------------------

class NeuralDStarLitePlanner(DStarLitePlanner):
    """
    D* Lite with a learned neural-network heuristic.

    On the first call to set_obstacles() the MLP is either loaded from a
    cached .npz file (keyed by obstacle hash) or trained from scratch using
    ground-truth costs collected by running vanilla D* Lite on random
    start/goal pairs.  Subsequent runs load instantly from cache.

    If training fails (e.g. degenerate map), falls back to Euclidean distance.
    """

    def __init__(self, resolution, map_width, map_height):
        super().__init__(resolution, map_width, map_height)
        self._nn   = None
        self._norm = math.sqrt(
            int(map_width  / resolution) ** 2 +
            int(map_height / resolution) ** 2
        )

    def set_obstacles(self, obstacle_list):
        super().set_obstacles(obstacle_list)
        from navigation.neural_heuristic import build_heuristic
        self._nn = build_heuristic(self.obstacles, self.width, self.height)

    def _h(self, a, b):
        if self._nn is None:
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        from navigation.neural_heuristic import encode_features, ADMISSIBILITY_FACTOR
        features = encode_features(a, b, self.obstacles, self.width, self.height)
        raw = float(self._nn.forward(features)) * self._norm
        return max(0.0, raw * ADMISSIBILITY_FACTOR)