import heapq
import math

from config import MAX_SPEED, MAX_STEERING, ROBOT_RADIUS


# ---------------------------------------------------------------------------
# Hybrid A* Planner  (kinodynamically feasible)
# ---------------------------------------------------------------------------

class HybridAStarPlanner:
    """
    Hybrid A* for a differential-drive robot.

    State   : (x, y, theta)  continuous position + heading
    Key     : (grid_x, grid_y, heading_bin)  for visited-state deduplication
    Motions : forward arcs generated from (MAX_SPEED, omega) pairs, where omega
              is sampled as fractions of MAX_STEERING — respecting the robot's
              minimum turning radius  R_min = MAX_SPEED / MAX_STEERING.
    Obstacles are pre-inflated by ROBOT_RADIUS (C-space approach) so the planner
    treats the robot as a point and collision checking is a single lookup.
    """

    N_HEADING   = 16                          # heading discretisation bins (~22.5° each)
    HEADING_RES = 2 * math.pi / N_HEADING

    # Forward arc primitives (omega as fraction of MAX_STEERING).
    _FWD_OMEGAS = [0.0,  1/3, -1/3,  2/3, -2/3,  1.0, -1.0]

    # Pivot-turn primitives: rotate one heading bin in place (v=0).
    # Cost is penalised so forward arcs are always preferred, but a pivot
    # always exists as a fallback so the search space stays fully connected.
    PIVOT_COST = 3.0   # cost per pivot step (vs self._step for a forward arc)

    INF = float('inf')

    # ------------------------------------------------------------------
    def __init__(self, resolution, map_width, map_height):
        self.resolution = resolution
        self.width      = int(map_width  / resolution)
        self.height     = int(map_height / resolution)
        self._inflated  = set()      # grid cells blocked after C-space inflation
        self._step      = resolution  # arc length per motion primitive (= one grid cell)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def world_to_grid(self, x, y):
        gx = int((x + self.width  * self.resolution / 2) / self.resolution)
        gy = int((y + self.height * self.resolution / 2) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        wx = gx * self.resolution - self.width  * self.resolution / 2
        wy = gy * self.resolution - self.height * self.resolution / 2
        return wx, wy

    # ------------------------------------------------------------------
    # C-space obstacle inflation
    # ------------------------------------------------------------------

    def set_obstacles(self, obstacle_list):
        """Inflate each obstacle cell by ROBOT_RADIUS using a circular mask."""
        raw = set()
        for ox, oy in obstacle_list:
            raw.add(self.world_to_grid(ox, oy))

        r = math.ceil(ROBOT_RADIUS / self.resolution)
        self._inflated.clear()
        for (gx, gy) in raw:
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= (r + 0.5) ** 2:
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            self._inflated.add((nx, ny))

    # ------------------------------------------------------------------
    # Kinematic helpers
    # ------------------------------------------------------------------

    def _hbin(self, theta):
        return int((theta % (2 * math.pi)) / self.HEADING_RES) % self.N_HEADING

    def _key(self, cx, cy, theta):
        gx, gy = self.world_to_grid(cx, cy)
        return (gx, gy, self._hbin(theta))

    def _propagate(self, cx, cy, theta, v, omega):
        """
        Simulate one arc of length self._step using differential-drive kinematics.
        v is signed: positive = forward, negative = reverse.
        The arc length is always self._step; dt is scaled by |v|.
        """
        dt = self._step / abs(v)
        if abs(omega) < 1e-9:
            return (cx + v * math.cos(theta) * dt,
                    cy + v * math.sin(theta) * dt,
                    theta)
        # Circular arc: R = v / omega  (negative R for reverse + positive omega)
        R      = v / omega
        ntheta = theta + omega * dt
        return (cx + R * (math.sin(ntheta) - math.sin(theta)),
                cy - R * (math.cos(ntheta) - math.cos(theta)),
                ntheta)

    def _free(self, cx, cy):
        gx, gy = self.world_to_grid(cx, cy)
        return (0 <= gx < self.width and
                0 <= gy < self.height and
                (gx, gy) not in self._inflated)

    def is_free(self, wx, wy, clearance=0):
        """
        Public check: True if the world position (wx, wy) is free of obstacles.
        clearance (in metres) adds an extra buffer on top of the standard
        ROBOT_RADIUS inflation — use this to keep goal points well away from walls.
        """
        gx, gy = self.world_to_grid(wx, wy)
        r = math.ceil(clearance / self.resolution)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= (r + 0.5) ** 2:
                    if (gx + dx, gy + dy) in self._inflated:
                        return False
        return 0 <= gx < self.width and 0 <= gy < self.height

    # ------------------------------------------------------------------
    # Hybrid A* search
    # ------------------------------------------------------------------

    def plan(self, start_pos, goal_pos, start_heading=None):
        sx, sy      = float(start_pos[0]), float(start_pos[1])
        gx_w, gy_w  = float(goal_pos[0]),  float(goal_pos[1])

        if start_heading is None:
            start_heading = math.atan2(gy_w - sy, gx_w - sx)

        # Temporarily remove start and goal grid cells from the inflated set
        # so a robot near a wall can still plan from/to its actual position.
        sg  = self.world_to_grid(sx, sy)
        gg  = self.world_to_grid(gx_w, gy_w)
        tmp = self._inflated & {sg, gg}
        self._inflated -= tmp

        goal_tol = self.resolution * 1.5

        def h(cx, cy):
            return math.hypot(cx - gx_w, cy - gy_w)

        # came_from[key] = (parent_key | None, self_cx, self_cy)
        # Storing each node's OWN position makes reconstruction straightforward.
        s_key     = self._key(sx, sy, start_heading)
        came_from = {s_key: (None, sx, sy)}
        g_score   = {s_key: 0.0}

        # heap: (f, g, tie-break, key, cx, cy, theta)
        tie  = 0
        heap = [(h(sx, sy), 0.0, tie, s_key, sx, sy, start_heading)]

        fwd_primitives = [(MAX_SPEED, frac * MAX_STEERING, self._step)
                          for frac in self._FWD_OMEGAS]

        result = []

        while heap:
            _, g, _, key, cx, cy, theta = heapq.heappop(heap)

            # Stale-entry check
            if g > g_score.get(key, self.INF) + 1e-9:
                continue

            if math.hypot(cx - gx_w, cy - gy_w) <= goal_tol:
                result = self._reconstruct(came_from, key, gx_w, gy_w)
                break

            # Forward arc expansions
            for v, omega, step_cost in fwd_primitives:
                nx, ny, ntheta = self._propagate(cx, cy, theta, v, omega)
                if not self._free(nx, ny):
                    continue
                nkey = self._key(nx, ny, ntheta)
                ng   = g + step_cost
                if ng < g_score.get(nkey, self.INF):
                    g_score[nkey]   = ng
                    came_from[nkey] = (key, nx, ny)
                    tie += 1
                    heapq.heappush(heap,
                                   (ng + h(nx, ny), ng, tie, nkey, nx, ny, ntheta))

            # Pivot-turn expansions (rotate one heading bin, no position change).
            # Only needed when the robot is wedged facing an obstacle — penalised
            # cost ensures forward arcs are always explored first.
            if self._free(cx, cy):
                for dtheta in (self.HEADING_RES, -self.HEADING_RES):
                    ntheta = theta + dtheta
                    nkey   = self._key(cx, cy, ntheta)
                    ng     = g + self.PIVOT_COST
                    if ng < g_score.get(nkey, self.INF):
                        g_score[nkey]   = ng
                        came_from[nkey] = (key, cx, cy)
                        tie += 1
                        heapq.heappush(heap,
                                       (ng + h(cx, cy), ng, tie, nkey, cx, cy, ntheta))

        # Restore temporarily cleared cells
        self._inflated |= tmp
        return result

    def _reconstruct(self, came_from, key, gx_w, gy_w):
        """Trace came_from back to start, return world-coordinate waypoint list."""
        path    = [(gx_w, gy_w)]
        cur_key = key
        while cur_key is not None:
            parent_key, cx, cy = came_from[cur_key]
            path.append((cx, cy))
            cur_key = parent_key
        path.reverse()

        # Remove consecutive duplicate positions produced by pivot-turn steps.
        # Keeping them would give zero-length segments whose atan2(0,0) heading
        # is always 0 (east), sending the controller in the wrong direction.
        deduped = [path[0]]
        for pt in path[1:]:
            if math.hypot(pt[0] - deduped[-1][0], pt[1] - deduped[-1][1]) > 1e-6:
                deduped.append(pt)
        return deduped


# ---------------------------------------------------------------------------
# D* Lite Planner  (holonomic, fast replanning)
# ---------------------------------------------------------------------------

class DStarLitePlanner:
    """
    D* Lite (Koenig & Likhachev, 2002).

    Plans from goal -> start so that replanning after obstacle changes is cheap.
    Treats the robot as a holonomic point — use HybridAStarPlanner when
    kinodynamic feasibility is required.
    The public interface matches HybridAStarPlanner: set_obstacles() + plan().
    """

    INF = float('inf')
    _MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def __init__(self, resolution, map_width, map_height):
        self.resolution = resolution
        self.width = int(map_width / resolution)
        self.height = int(map_height / resolution)
        self.obstacles = set()

        self._g_map   = {}
        self._rhs_map = {}
        self._open_list = []
        self._open_dict = {}
        self._km    = 0
        self._start = None
        self._goal  = None

    # ------------------------------------------------------------------
    # Coordinate helpers
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

    def _key(self, s):
        min_val = min(self._g_map.get(s, self.INF), self._rhs_map.get(s, self.INF))
        return (min_val + self._h(self._start, s) + self._km, min_val)

    def _neighbors(self, s):
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

        self._open_dict.pop(s, None)

        if self._g_map.get(s, self.INF) != self._rhs_map.get(s, self.INF):
            self._push(s, self._key(s))

    def _compute_shortest_path(self):
        while self._open_list:
            k_old, s = heapq.heappop(self._open_list)

            if self._open_dict.get(s) != k_old:
                continue

            k_new     = self._key(s)
            start_key = self._key(self._start)

            if k_old >= start_key and self._rhs_map.get(self._start, self.INF) == self._g_map.get(self._start, self.INF):
                break

            if k_old < k_new:
                self._push(s, k_new)
            elif self._g_map.get(s, self.INF) > self._rhs_map.get(s, self.INF):
                self._g_map[s] = self._rhs_map[s]
                del self._open_dict[s]
                for sp, _ in self._neighbors(s):
                    self._update_vertex(sp)
            else:
                self._g_map[s] = self.INF
                self._update_vertex(s)
                for sp, _ in self._neighbors(s):
                    self._update_vertex(sp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, start_pos, goal_pos, start_heading=None):
        sx, sy = self.world_to_grid(start_pos[0], start_pos[1])
        gx, gy = self.world_to_grid(goal_pos[0], goal_pos[1])

        self.obstacles.discard((sx, sy))
        self.obstacles.discard((gx, gy))

        self._start   = (sx, sy)
        self._goal    = (gx, gy)
        self._km      = 0
        self._g_map   = {}
        self._rhs_map = {self._goal: 0.0}
        self._open_list = []
        self._open_dict = {}

        self._push(self._goal, self._key(self._goal))
        self._compute_shortest_path()

        if self._g_map.get(self._start, self.INF) == self.INF:
            return []

        path    = []
        current = self._start
        visited = set()
        while current != self._goal:
            if current in visited:
                return []
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
# A* Planner  (holonomic baseline)
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
        self.obstacles.clear()
        for (ox, oy) in obstacle_list:
            gx, gy = self.world_to_grid(ox, oy)
            self.obstacles.add((gx, gy))

    def world_to_grid(self, x, y):
        gx = int((x + (self.width * self.resolution) / 2) / self.resolution)
        gy = int((y + (self.height * self.resolution) / 2) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        wx = (gx * self.resolution) - (self.width * self.resolution) / 2
        wy = (gy * self.resolution) - (self.height * self.resolution) / 2
        return wx, wy

    def plan(self, start_pos, goal_pos, start_heading=None):
        sx, sy = self.world_to_grid(start_pos[0], start_pos[1])
        gx, gy = self.world_to_grid(goal_pos[0], goal_pos[1])

        if (sx, sy) in self.obstacles: self.obstacles.remove((sx, sy))
        if (gx, gy) in self.obstacles: self.obstacles.remove((gx, gy))

        start_node = Node(sx, sy, 0, self.heuristic(sx, sy, gx, gy))
        open_list  = []
        heapq.heappush(open_list, start_node)
        visited = {(sx, sy)}

        while open_list:
            current = heapq.heappop(open_list)

            if abs(current.x - gx) <= 1 and abs(current.y - gy) <= 1:
                return self.reconstruct_path(current)

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny    = current.x + dx, current.y + dy
                move_cost = math.sqrt(dx ** 2 + dy ** 2)

                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) not in self.obstacles and (nx, ny) not in visited:
                        new_node = Node(nx, ny,
                                        current.cost + move_cost,
                                        self.heuristic(nx, ny, gx, gy),
                                        current)
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
        return path[::-1]
