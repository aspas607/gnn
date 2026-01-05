# formation_path_planner.py
"""
简单的基于网格的 A* 路径规划器，用于给编队提供全局路径（避开圆形障碍物）
"""
import heapq
import numpy as np
from typing import List, Tuple
from obstacle_environment_2d import ObstacleEnvironment2D


def world_to_grid(pos: np.ndarray, env_size: float, resolution: float) -> Tuple[int, int]:
    grid_size = int(np.ceil(env_size / resolution))
    x_idx = int(np.clip(pos[0] / resolution, 0, grid_size - 1))
    y_idx = int(np.clip(pos[1] / resolution, 0, grid_size - 1))
    return x_idx, y_idx


def grid_to_world(idx: Tuple[int, int], resolution: float) -> np.ndarray:
    return np.array([(idx[0] + 0.5) * resolution, (idx[1] + 0.5) * resolution])


def build_occupancy_grid(env: ObstacleEnvironment2D, resolution: float = 2.0) -> Tuple[np.ndarray, float]:
    env_size = env.env_size
    grid_size = int(np.ceil(env_size / resolution))
    occ = np.zeros((grid_size, grid_size), dtype=np.uint8)

    # 标注被障碍物覆盖的格子
    for i in range(grid_size):
        for j in range(grid_size):
            world_pos = grid_to_world((i, j), resolution)
            if env.check_collision(world_pos, agent_radius=0.5):
                occ[i, j] = 1
    return occ, resolution


def astar(occ: np.ndarray, start_idx: Tuple[int, int], goal_idx: Tuple[int, int]) -> List[Tuple[int, int]]:
    # 8连通 A*
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    h = lambda a, b: np.hypot(a[0] - b[0], a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0 + h(start_idx, goal_idx), 0, start_idx))
    came_from = {}
    g_score = {start_idx: 0}

    max_x, max_y = occ.shape

    while open_set:
        _, g_curr, current = heapq.heappop(open_set)
        if current == goal_idx:
            # 重建路径
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy in neighbors:
            nx, ny = current[0] + dx, current[1] + dy
            if nx < 0 or ny < 0 or nx >= max_x or ny >= max_y:
                continue
            if occ[nx, ny]:
                continue
            tentative_g = g_curr + np.hypot(dx, dy)
            neighbor = (nx, ny)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + h(neighbor, goal_idx)
                heapq.heappush(open_set, (f, tentative_g, neighbor))
    return []


def smooth_path(path: List[np.ndarray], step: int = 3) -> List[np.ndarray]:
    if len(path) <= 2:
        return path
    # 简单的降采样平滑
    return [path[i] for i in range(0, len(path), step)] + [path[-1]]


def plan_path(env: ObstacleEnvironment2D,
              start: np.ndarray,
              goal: np.ndarray,
              resolution: float = 2.0) -> List[np.ndarray]:
    """
    在给定环境中对(start->goal)进行基于网格的A*路径规划，返回世界坐标下的路径点列表
    """
    occ, res = build_occupancy_grid(env, resolution=resolution)
    start_idx = world_to_grid(start, env.env_size, res)
    goal_idx = world_to_grid(goal, env.env_size, res)

    path_idx = astar(occ, start_idx, goal_idx)
    if not path_idx:
        return []

    path_world = [grid_to_world(idx, res) for idx in path_idx]
    path_smooth = smooth_path(path_world, step=max(1, int(4.0 / (res / 2.0))))
    return path_smooth


if __name__ == "__main__":
    # 简单示例：在命令行上测试规划
    env = ObstacleEnvironment2D(env_size=300.0, num_obstacles=5, seed=42)
    start = np.array([15.0, 15.0])
    goal = np.array([285.0, 285.0])
    path = plan_path(env, start, goal, resolution=3.0)
    print(f"Planned {len(path)} waypoints")
