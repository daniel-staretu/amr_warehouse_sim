import heapq
import math


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