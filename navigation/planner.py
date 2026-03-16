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
        self.resolution      = resolution
        self.width           = int(map_width  / resolution)
        self.height          = int(map_height / resolution)
        self._static_inflated  = set()   # cells from set_obstacles (never removed at runtime)
        self._dynamic_inflated = set()   # cells from add_dynamic_obstacle (removable)
        self._inflated         = set()   # union — what the planner actually checks
        self._step             = resolution

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
        self._static_inflated.clear()
        for (gx, gy) in raw:
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= (r + 0.5) ** 2:
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            self._static_inflated.add((nx, ny))
        self._inflated = self._static_inflated | self._dynamic_inflated

    # ------------------------------------------------------------------
    # Dynamic obstacle management
    # ------------------------------------------------------------------

    def _inflate_cell(self, gx, gy):
        """Return the set of C-space cells produced by inflating grid cell (gx, gy)."""
        r = math.ceil(ROBOT_RADIUS / self.resolution)
        cells = set()
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= (r + 0.5) ** 2:
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        cells.add((nx, ny))
        return cells

    def add_dynamic_obstacle(self, wx, wy):
        """
        Register a newly detected dynamic obstacle at world position (wx, wy).
        Inflates by ROBOT_RADIUS and adds to the live cost map immediately.
        Call plan() again after adding obstacles to get a replanned path.
        """
        gx, gy = self.world_to_grid(wx, wy)
        new_cells = self._inflate_cell(gx, gy)
        self._dynamic_inflated |= new_cells
        self._inflated         |= new_cells

    def remove_dynamic_obstacles(self):
        """
        Clear all runtime-added obstacles, restoring the static-only cost map.
        Useful at the start of each replanning cycle when using an obstacle
        tracker that provides a complete fresh set of detections each frame.
        """
        self._dynamic_inflated.clear()
        self._inflated = set(self._static_inflated)

    # ------------------------------------------------------------------
    # Path validation — replanning trigger
    # ------------------------------------------------------------------

    def _segment_free(self, a, b, obstacle_set):
        """
        Bresenham ray-cast from grid cell a to b.
        Returns True if no cell in obstacle_set lies on the segment.
        """
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x1 >= x0 else -1
        sy = 1 if y1 >= y0 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            if (x, y) in obstacle_set:
                return False
            if x == x1 and y == y1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def path_is_blocked(self, path):
        """
        Return True if the current path passes through any dynamic obstacle cell.

        Checks are made against _dynamic_inflated only — static obstacles were
        already avoided during planning so they never trigger a false positive.

        Both waypoint positions and the straight-line segments between them are
        tested (Bresenham), so an obstacle sitting between two waypoints is caught
        even when it doesn't coincide exactly with a waypoint.
        """
        if not self._dynamic_inflated:
            return False   # fast-path: no dynamic obstacles present

        for i, wp in enumerate(path):
            gx, gy = self.world_to_grid(wp[0], wp[1])
            if (gx, gy) in self._dynamic_inflated:
                return True
            if i < len(path) - 1:
                a = self.world_to_grid(wp[0], wp[1])
                b = self.world_to_grid(path[i + 1][0], path[i + 1][1])
                if not self._segment_free(a, b, self._dynamic_inflated):
                    return True
        return False

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
# Kinodynamic D* Lite Planner
# ---------------------------------------------------------------------------

class KinoDStarLitePlanner(DStarLitePlanner):
    """
    D* Lite extended with kinodynamic-quality path post-processing.

    Stages applied after the base grid search:
      1. C-space inflation   — identical to HybridAStarPlanner, so the robot
                               centre never enters an obstacle cell.
      2. Theta* smoothing    — Bresenham line-of-sight pass removes unnecessary
                               grid-aligned turns, giving any-angle straight legs.
      3. Turning-radius arcs — sharp corners are replaced with circular arc
                               samples so no heading change requires a tighter
                               turn than R_min = MAX_SPEED / MAX_STEERING.
      4. Path validation     — every waypoint is confirmed free; blocked points
                               fall back to the nearest raw grid waypoint.
      5. Heading annotation  — each waypoint is returned as (x, y, theta) where
                               theta points toward the next waypoint.

    Output: list of (x, y, theta) tuples — compatible with PurePursuitController
    without any changes to controller.py.

    Dynamic-obstacle interface
    --------------------------
    update_dynamic_obstacles(dynamic_obs) ingests new (x, y) detections (e.g.
    from a YOLO detector) into the inflated cost map.  Call plan() afterwards
    to get a replanned path.  The incremental D* Lite update mechanism is
    scaffolded here and marked TODO for full implementation.
    """

    def __init__(self, resolution, map_width, map_height):
        super().__init__(resolution, map_width, map_height)
        self._static_inflated  = set()
        self._dynamic_inflated = set()
        self._inflated         = set()

    # ------------------------------------------------------------------
    # Obstacle management
    # ------------------------------------------------------------------

    def set_obstacles(self, obstacle_list):
        """Build inflated C-space from the static obstacle list."""
        raw = set()
        for ox, oy in obstacle_list:
            raw.add(self.world_to_grid(ox, oy))

        r = math.ceil(ROBOT_RADIUS / self.resolution)
        self._static_inflated.clear()
        for gx, gy in raw:
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= (r + 0.5) ** 2:
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            self._static_inflated.add((nx, ny))

        self._inflated = self._static_inflated | self._dynamic_inflated
        self.obstacles = set(self._inflated)

    def is_free(self, wx, wy, clearance=0):
        """True if (wx, wy) is outside all inflated obstacle cells."""
        gx, gy = self.world_to_grid(wx, wy)
        r = math.ceil(clearance / self.resolution)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= (r + 0.5) ** 2:
                    if (gx + dx, gy + dy) in self._inflated:
                        return False
        return 0 <= gx < self.width and 0 <= gy < self.height

    def _inflate_cell(self, gx, gy):
        """Return C-space cells produced by inflating grid cell (gx, gy)."""
        r = math.ceil(ROBOT_RADIUS / self.resolution)
        cells = set()
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= (r + 0.5) ** 2:
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        cells.add((nx, ny))
        return cells

    def add_dynamic_obstacle(self, wx, wy):
        """
        Register a newly detected dynamic obstacle at world position (wx, wy).
        Inflates by ROBOT_RADIUS and adds to the live cost map immediately.
        Call plan() again after adding obstacles to get a replanned path.

        TODO: wire into D* Lite's incremental vertex-update mechanism
              so only the affected search-tree nodes are repaired.
        """
        gx, gy = self.world_to_grid(wx, wy)
        new_cells = self._inflate_cell(gx, gy)
        self._dynamic_inflated |= new_cells
        self._inflated         |= new_cells
        self.obstacles          = set(self._inflated)

    def remove_dynamic_obstacles(self):
        """
        Clear all runtime-added obstacles, restoring the static-only cost map.
        Call before ingesting a fresh set of detections each frame.
        """
        self._dynamic_inflated.clear()
        self._inflated = set(self._static_inflated)
        self.obstacles = set(self._inflated)

    # ------------------------------------------------------------------
    # Path validation — replanning trigger
    # ------------------------------------------------------------------

    def _segment_free(self, a, b, obstacle_set):
        """Bresenham ray-cast from a to b; True if no cell in obstacle_set is hit."""
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x1 >= x0 else -1
        sy = 1 if y1 >= y0 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            if (x, y) in obstacle_set:
                return False
            if x == x1 and y == y1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def path_is_blocked(self, path):
        """
        Return True if the current path passes through any dynamic obstacle cell.
        Checks waypoint positions and Bresenham segments between them.
        """
        if not self._dynamic_inflated:
            return False
        for i, wp in enumerate(path):
            gx, gy = self.world_to_grid(wp[0], wp[1])
            if (gx, gy) in self._dynamic_inflated:
                return True
            if i < len(path) - 1:
                a = self.world_to_grid(wp[0], wp[1])
                b = self.world_to_grid(path[i + 1][0], path[i + 1][1])
                if not self._segment_free(a, b, self._dynamic_inflated):
                    return True
        return False

    # ------------------------------------------------------------------
    # Stage 1 — Theta* any-angle smoothing
    # ------------------------------------------------------------------

    def _line_of_sight(self, a, b):
        """
        Bresenham ray-cast from grid cell a to grid cell b on the inflated grid.
        Returns True if every cell along the segment is free.
        """
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x1 >= x0 else -1
        sy = 1 if y1 >= y0 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            if (x, y) in self._inflated:
                return False
            if x == x1 and y == y1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _smooth_path(self, grid_path):
        """
        Theta* greedy pass: repeatedly anchor at the last kept waypoint and
        skip forward as far as line-of-sight holds, yielding any-angle segments.
        """
        if len(grid_path) <= 2:
            return grid_path
        smoothed = [grid_path[0]]
        i = 0
        while i < len(grid_path) - 1:
            j = len(grid_path) - 1
            while j > i + 1:
                if self._line_of_sight(smoothed[-1], grid_path[j]):
                    break
                j -= 1
            smoothed.append(grid_path[j])
            i = j
        return smoothed

    # ------------------------------------------------------------------
    # Stage 2 — Heading annotation
    # ------------------------------------------------------------------

    def _add_headings(self, world_path, start_heading):
        """
        Convert a list of (x, y) waypoints into (x, y, theta) tuples where
        theta is the heading toward the next waypoint.  The last waypoint
        inherits the heading of the final segment.
        """
        if not world_path:
            return []
        result = []
        for i in range(len(world_path) - 1):
            x,  y  = world_path[i]
            nx, ny = world_path[i + 1]
            result.append((x, y, math.atan2(ny - y, nx - x)))
        last = world_path[-1]
        last_theta = result[-1][2] if result else (start_heading or 0.0)
        result.append((last[0], last[1], last_theta))
        return result

    # ------------------------------------------------------------------
    # Stage 3 — Turning-radius enforcement
    # ------------------------------------------------------------------

    def _enforce_turning_radius(self, world_path, min_radius):
        """
        Walk the smoothed path and replace any corner whose required turning
        radius is tighter than min_radius with a circular arc sampled at
        resolution * 0.5 m intervals.  Arc points that land inside a blocked
        cell are silently skipped so obstacle avoidance is never compromised.
        """
        if len(world_path) <= 2 or min_radius <= 0:
            return world_path

        ARC_STEP = self.resolution * 0.5
        result = [world_path[0]]

        for i in range(1, len(world_path) - 1):
            A = result[-1]
            B = world_path[i]
            C = world_path[i + 1]

            dist_AB = math.hypot(B[0] - A[0], B[1] - A[1])
            dist_BC = math.hypot(C[0] - B[0], C[1] - B[1])
            if dist_AB < 1e-6 or dist_BC < 1e-6:
                result.append(B)
                continue

            h1 = math.atan2(B[1] - A[1], B[0] - A[0])
            h2 = math.atan2(C[1] - B[1], C[0] - B[0])
            delta = (h2 - h1 + math.pi) % (2 * math.pi) - math.pi

            if abs(delta) < 1e-4:
                result.append(B)
                continue

            # Tangent offset d = R * tan(|delta| / 2), clamped to half each leg
            d = min_radius * abs(math.tan(delta / 2))
            d = min(d, dist_AB * 0.5, dist_BC * 0.5)
            if d < 1e-3:
                result.append(B)
                continue

            # Tangent points on the incoming / outgoing legs
            t1 = (B[0] - d * math.cos(h1), B[1] - d * math.sin(h1))
            t2 = (B[0] + d * math.cos(h2), B[1] + d * math.sin(h2))

            # Arc centre: perpendicular to h1 at T1 (sign follows turn direction)
            sign = math.copysign(1.0, delta)
            perp = h1 + sign * math.pi / 2
            arc_cx = t1[0] + min_radius * math.cos(perp)
            arc_cy = t1[1] + min_radius * math.sin(perp)

            a_start = math.atan2(t1[1] - arc_cy, t1[0] - arc_cx)
            a_end   = math.atan2(t2[1] - arc_cy, t2[0] - arc_cx)
            arc_span = (a_end - a_start + math.pi) % (2 * math.pi) - math.pi
            if arc_span * delta < 0:
                arc_span += sign * 2 * math.pi

            n_steps = max(2, int(math.ceil(min_radius * abs(arc_span) / ARC_STEP)))
            for k in range(n_steps + 1):
                angle = a_start + (k / n_steps) * arc_span
                px = arc_cx + min_radius * math.cos(angle)
                py = arc_cy + min_radius * math.sin(angle)
                gx, gy = self.world_to_grid(px, py)
                if ((gx, gy) not in self._inflated and
                        0 <= gx < self.width and 0 <= gy < self.height):
                    result.append((px, py))

        result.append(world_path[-1])
        return result

    # ------------------------------------------------------------------
    # Stage 4 — Path validation
    # ------------------------------------------------------------------

    def _validate_path(self, smoothed_world, raw_world):
        """
        Confirm every waypoint is outside the inflated obstacle set.
        Any blocked waypoint is replaced with the nearest raw grid waypoint
        and a warning is printed, so the path always degrades gracefully.
        """
        validated = []
        for wx, wy in smoothed_world:
            gx, gy = self.world_to_grid(wx, wy)
            if ((gx, gy) in self._inflated or
                    not (0 <= gx < self.width and 0 <= gy < self.height)):
                print(f"Warning [KinoDStarLite]: waypoint ({wx:.2f}, {wy:.2f}) "
                      f"is inside an obstacle — falling back to raw segment.")
                best = min(raw_world,
                           key=lambda p: math.hypot(p[0] - wx, p[1] - wy))
                validated.append(best)
            else:
                validated.append((wx, wy))
        return validated

    # ------------------------------------------------------------------
    # Public plan() — orchestrates all post-processing stages
    # ------------------------------------------------------------------

    def plan(self, start_pos, goal_pos, start_heading=None):
        # Restore a clean obstacle set (base plan() permanently discards
        # the start / goal cells from self.obstacles on each call).
        self.obstacles = set(self._inflated)

        raw_world = super().plan(start_pos, goal_pos, start_heading)
        if not raw_world:
            return []

        # Stage 1 — Theta* any-angle smoothing (operates in grid space)
        grid_path     = [self.world_to_grid(wx, wy) for wx, wy in raw_world]
        smoothed_grid = self._smooth_path(grid_path)
        smoothed_world = [self.grid_to_world(gx, gy) for gx, gy in smoothed_grid]

        # Stage 2 — Validate smoothed waypoints; patch with raw fallback
        smoothed_world = self._validate_path(smoothed_world, raw_world)

        # Stage 3 — Insert arc waypoints at corners tighter than R_min
        min_radius = MAX_SPEED / MAX_STEERING
        smoothed_world = self._enforce_turning_radius(smoothed_world, min_radius)

        # Stage 4 — Annotate with headings → (x, y, theta)
        return self._add_headings(smoothed_world, start_heading)


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
