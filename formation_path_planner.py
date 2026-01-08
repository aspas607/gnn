# formation_path_planner.py
"""
BIT*（Batch Informed Trees）全局路径规划器（非强化学习），用于给编队提供高质量的连续空间全局路径。

为什么选 BIT*：
- 面向“最短路径/最小代价”的几何规划，BIT* 是经典的 anytime + 渐近最优（asymptotically optimal）采样规划方法；
- 它用 A* 风格的启发式顺序搜索隐式随机几何图（RGG），通常比 RRT* / Informed RRT* 更快收敛到更优解（论文实验结论）。

实现特点（适配你当前工程）：
- 仅依赖 numpy；碰撞检测完全通过 env.check_collision(点, agent_radius=...) 这个黑盒接口
- 连续空间规划：不走格子，不依赖占据栅格分辨率
- 输出后处理：捷径平滑（shortcut）+ 等间距重采样，更利于下游跟踪控制
- 仅保留 BIT* 一种方法（无 A* / RRT* 回退），符合你“只保留最新方法”的要求

接口保持兼容：plan_path(env, start, goal, resolution=...)
- resolution 在这里用于“碰撞检测线段离散步长”和“路径重采样间距”的参考尺度（不是网格分辨率）
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from obstacle_environment_2d import ObstacleEnvironment2D


# -----------------------------
# 基础几何/碰撞检测
# -----------------------------
def _clamp_to_bounds(p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.array([float(np.clip(p[0], lo, hi)), float(np.clip(p[1], lo, hi))], dtype=float)


def _segment_is_free(env: ObstacleEnvironment2D,
                     a: np.ndarray,
                     b: np.ndarray,
                     *,
                     agent_radius: float,
                     step: float) -> bool:
    """
    线段离散采样碰撞检测（黑盒 env.check_collision）。
    step 越小越保守，但更慢。通常取 ~resolution 量级即可。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b - a
    dist = float(np.linalg.norm(d))
    if dist <= 1e-9:
        return not env.check_collision(a, agent_radius=agent_radius)

    n = max(2, int(math.ceil(dist / max(step, 1e-6))) + 1)
    for t in np.linspace(0.0, 1.0, n):
        p = a + t * d
        if env.check_collision(p, agent_radius=agent_radius):
            return False
    return True


def _nudge_to_free(env: ObstacleEnvironment2D,
                   p: np.ndarray,
                   *,
                   agent_radius: float,
                   bounds: Tuple[float, float],
                   nudge_scale: float,
                   rng: random.Random,
                   tries: int = 120) -> Optional[np.ndarray]:
    """
    若 start/goal 落在障碍内，尝试轻微抖动到可行位置。
    """
    lo, hi = bounds
    p = _clamp_to_bounds(p, lo, hi)
    if not env.check_collision(p, agent_radius=agent_radius):
        return p

    for _ in range(tries):
        jitter = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)], dtype=float)
        q = _clamp_to_bounds(p + jitter * nudge_scale, lo, hi)
        if not env.check_collision(q, agent_radius=agent_radius):
            return q
    return None


def _heuristic_to_goal(p: np.ndarray, goal: np.ndarray) -> float:
    return float(np.linalg.norm(goal - p))


# -----------------------------
# Informed 采样（椭球）
# -----------------------------
def _sample_unit_ball_2d(rng: random.Random) -> np.ndarray:
    # 均匀采样单位圆内点
    theta = rng.random() * 2.0 * math.pi
    radius = math.sqrt(rng.random())
    return np.array([radius * math.cos(theta), radius * math.sin(theta)], dtype=float)


def _informed_sample_ellipsoid(start: np.ndarray,
                               goal: np.ndarray,
                               c_best: float,
                               rng: random.Random) -> np.ndarray:
    """
    以 start/goal 为焦点的椭球内采样（Informed set）。
    仅当已有可行路径（c_best 有限且 > c_min）时使用。
    """
    c_min = float(np.linalg.norm(goal - start))
    if not math.isfinite(c_best) or c_best <= c_min + 1e-9:
        raise ValueError("No informed set yet")

    a = c_best / 2.0
    b = math.sqrt(max(c_best * c_best - c_min * c_min, 1e-12)) / 2.0

    x_ball = _sample_unit_ball_2d(rng)
    x_ell = np.array([a * x_ball[0], b * x_ball[1]], dtype=float)

    d = goal - start
    theta = math.atan2(d[1], d[0])
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=float)

    center = (start + goal) / 2.0
    return (R @ x_ell) + center


# -----------------------------
# BIT* 数据结构
# -----------------------------
@dataclass
class _Vertex:
    pos: np.ndarray
    parent: int
    g: float


@dataclass(frozen=True)
class _Edge:
    v: int              # from-vertex index
    target_is_sample: bool
    target_id: int      # sample id if target_is_sample else vertex index


# -----------------------------
# 路径后处理：捷径平滑 + 重采样
# -----------------------------
def shortcut_smooth(env: ObstacleEnvironment2D,
                    path: List[np.ndarray],
                    *,
                    agent_radius: float,
                    collision_step: float,
                    iters: int = 220,
                    seed: int = 0) -> List[np.ndarray]:
    if len(path) <= 2:
        return path
    rng = random.Random(seed)
    pts = [p.copy() for p in path]

    for _ in range(iters):
        if len(pts) <= 2:
            break
        i = rng.randrange(0, len(pts) - 1)
        j = rng.randrange(i + 1, len(pts))
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        if _segment_is_free(env, a, b, agent_radius=agent_radius, step=collision_step):
            pts = pts[:i + 1] + pts[j:]
    return pts


def resample_path(path: List[np.ndarray], spacing: float = 3.0) -> List[np.ndarray]:
    if len(path) <= 1:
        return path
    spacing = max(float(spacing), 1e-6)

    out = [path[0].copy()]
    acc = 0.0

    for k in range(1, len(path)):
        a = out[-1]
        b = path[k]
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 1e-9:
            continue

        direction = seg / seg_len
        dist_left = seg_len
        cur = a.copy()

        while dist_left > 0.0:
            need = spacing - acc
            if dist_left >= need:
                cur = cur + direction * need
                out.append(cur.copy())
                dist_left -= need
                acc = 0.0
            else:
                acc += dist_left
                dist_left = 0.0

    if float(np.linalg.norm(out[-1] - path[-1])) > 1e-6:
        out.append(path[-1].copy())
    return out


# -----------------------------
# BIT* 主算法（2D，最短路）
# -----------------------------
def _compute_connection_radius(env_size: float,
                               n: int,
                               *,
                               r_min: float,
                               r_max: float,
                               scale: float = 1.2) -> float:
    """
    BIT*/RGG 的近邻半径（工程化版本）。
    n 取 |V|+|X| 的规模；二维空间下半径随 n 增大而收缩。
    """
    diag = env_size * math.sqrt(2.0)
    n = max(int(n), 2)
    r = scale * diag * math.sqrt(math.log(n + 1.0) / (n + 1.0))
    return float(np.clip(r, r_min, r_max))


def _reconstruct_path(vertices: List[_Vertex], goal_vid: int) -> List[np.ndarray]:
    path: List[np.ndarray] = []
    cur = goal_vid
    while cur != -1:
        path.append(vertices[cur].pos.copy())
        cur = vertices[cur].parent
    path.reverse()
    return path


def _propagate_costs_from(vertices: List[_Vertex],
                          children: List[List[int]],
                          root: int) -> None:
    """
    rewiring 后，把 root 子树的 g 代价一致性更新。
    """
    stack = [root]
    while stack:
        v = stack.pop()
        pv = vertices[v].parent
        if pv != -1:
            vertices[v].g = vertices[pv].g + float(np.linalg.norm(vertices[v].pos - vertices[pv].pos))
        for c in children[v]:
            stack.append(c)


def _bitstar_plan(env: ObstacleEnvironment2D,
                  start: np.ndarray,
                  goal: np.ndarray,
                  *,
                  agent_radius: float,
                  bounds: Tuple[float, float],
                  resolution: float,
                  max_samples: int,
                  batch_size: int,
                  seed: int,
                  goal_sample_rate: float = 0.05) -> List[np.ndarray]:
    """
    返回最优路径（找到的最好解）；找不到返回 []
    """
    lo, hi = bounds
    rng = random.Random(seed)

    # 先确保 start/goal 可行
    nudge_scale = max(2.0, float(resolution) * 2.0)
    s = _nudge_to_free(env, start, agent_radius=agent_radius, bounds=bounds, nudge_scale=nudge_scale, rng=rng)
    g = _nudge_to_free(env, goal, agent_radius=agent_radius, bounds=bounds, nudge_scale=nudge_scale, rng=rng)
    if s is None or g is None:
        return []
    start = s
    goal = g

    vertices: List[_Vertex] = [_Vertex(pos=start.copy(), parent=-1, g=0.0)]
    children: List[List[int]] = [[]]

    # samples 用 dict：id -> pos
    next_sid = 0
    samples: Dict[int, np.ndarray] = {}

    def add_samples(num: int, c_best: float) -> None:
        nonlocal next_sid
        for _ in range(num):
            # 少量直接采 goal，提升首次可行解速度（BIT* 也常用 goal bias）
            if rng.random() < goal_sample_rate:
                p = goal.copy()
            else:
                if math.isfinite(c_best):
                    try:
                        p = _informed_sample_ellipsoid(start, goal, c_best, rng)
                    except ValueError:
                        p = np.array([rng.uniform(lo, hi), rng.uniform(lo, hi)], dtype=float)
                else:
                    p = np.array([rng.uniform(lo, hi), rng.uniform(lo, hi)], dtype=float)

            p = _clamp_to_bounds(np.asarray(p, dtype=float), lo, hi)
            # 采样点不必都可行（BIT* 会在边验证阶段碰撞剔除），但这里过滤一下可提高效率
            if env.check_collision(p, agent_radius=agent_radius):
                continue
            samples[next_sid] = p
            next_sid += 1

    # 总是把 goal 作为一个可连通的样本（避免“永远差一点连不到”）
    samples[next_sid] = goal.copy()
    goal_sid = next_sid
    next_sid += 1

    # 启发式：到 goal 的距离
    def h(pos: np.ndarray) -> float:
        return _heuristic_to_goal(pos, goal)

    # edge collision cache：避免重复线段检测
    edge_free_cache: Dict[Tuple[int, bool, int], bool] = {}

    # 全局最优解记录
    best_cost = float("inf")
    best_goal_vid: Optional[int] = None

    # 队列：顶点扩展 Qv；边评估 Qe
    # 优先级都是“对最优解的下界估计”
    Qv: List[Tuple[float, int]] = []  # (f_hat(v), v)
    Qe: List[Tuple[float, _Edge]] = []  # (f_hat(edge), edge)

    # 初始采样
    add_samples(batch_size, best_cost)

    # 主循环：分批次增密隐式 RGG
    samples_budget = max(1, int(max_samples))
    while (len(vertices) + len(samples)) < samples_budget:
        # 连接半径
        n_total = len(vertices) + len(samples)
        r = _compute_connection_radius(float(env.env_size), n_total,
                                       r_min=max(4.0, float(resolution) * 2.0),
                                       r_max=float(env.env_size) * 0.6,
                                       scale=1.2)

        # 重置队列：每批次都从“当前树 + 当前样本”重新启发式有序搜索
        Qv.clear()
        Qe.clear()

        for vid, vtx in enumerate(vertices):
            # 顶点的潜在最优 f 下界
            heapq.heappush(Qv, (vtx.g + h(vtx.pos), vid))

        # batch search
        while (Qv or Qe):
            best_qv = Qv[0][0] if Qv else float("inf")
            best_qe = Qe[0][0] if Qe else float("inf")

            # 如果最乐观下界都不可能改进当前最优解，结束本批次搜索，扩充样本
            if min(best_qv, best_qe) >= best_cost:
                break

            if best_qe <= best_qv:
                # 扩展边
                _, edge = heapq.heappop(Qe)
                v = edge.v

                # target position
                if edge.target_is_sample:
                    if edge.target_id not in samples:
                        continue  # 已被吸纳/删除
                    x_pos = samples[edge.target_id]
                    x_is_goal = (edge.target_id == goal_sid)
                else:
                    if edge.target_id >= len(vertices):
                        continue
                    x_pos = vertices[edge.target_id].pos
                    x_is_goal = False

                v_pos = vertices[v].pos
                g_v = vertices[v].g
                c_vx = float(np.linalg.norm(x_pos - v_pos))

                # 再次剪枝
                if (g_v + c_vx + h(x_pos)) >= best_cost:
                    continue

                # 懒碰撞检测 + cache
                cache_key = (v, edge.target_is_sample, edge.target_id)
                ok = edge_free_cache.get(cache_key, None)
                if ok is None:
                    ok = _segment_is_free(env, v_pos, x_pos, agent_radius=agent_radius,
                                          step=max(0.5, float(resolution) * 0.8))
                    edge_free_cache[cache_key] = ok
                if not ok:
                    continue

                new_g = g_v + c_vx

                if edge.target_is_sample:
                    # 将样本提升为树顶点
                    new_vid = len(vertices)
                    vertices.append(_Vertex(pos=x_pos.copy(), parent=v, g=new_g))
                    children.append([])
                    children[v].append(new_vid)
                    del samples[edge.target_id]

                    # 若连接到 goal，则更新最优解
                    if x_is_goal and new_g < best_cost:
                        best_cost = new_g
                        best_goal_vid = new_vid

                else:
                    # 可能的 rewiring（改父节点）
                    u = edge.target_id
                    if new_g + 1e-9 < vertices[u].g:
                        # 从旧父节点 children 中移除
                        old_p = vertices[u].parent
                        if old_p != -1:
                            try:
                                children[old_p].remove(u)
                            except ValueError:
                                pass
                        vertices[u].parent = v
                        children[v].append(u)
                        # 更新子树代价
                        _propagate_costs_from(vertices, children, u)

                        # 若 u 是当前 goal 顶点（理论上 goal 会变成顶点），可更新最优
                        if best_goal_vid == u and vertices[u].g < best_cost:
                            best_cost = vertices[u].g

            else:
                # 扩展顶点：生成候选边（到近邻样本/顶点）
                _, v = heapq.heappop(Qv)
                v_pos = vertices[v].pos
                g_v = vertices[v].g

                # 近邻样本
                if samples:
                    sids = list(samples.keys())
                    spos = np.vstack([samples[sid] for sid in sids])
                    d = np.linalg.norm(spos - v_pos[None, :], axis=1)
                    near = np.where(d <= r)[0]
                    for idx in near.tolist():
                        sid = sids[idx]
                        x_pos = samples[sid]
                        f_hat = g_v + float(d[idx]) + h(x_pos)
                        if f_hat < best_cost:
                            heapq.heappush(Qe, (f_hat, _Edge(v=v, target_is_sample=True, target_id=sid)))

                # 近邻顶点（用于 rewiring）
                if len(vertices) > 1:
                    vpos_all = np.vstack([vx.pos for vx in vertices])
                    d2 = np.linalg.norm(vpos_all - v_pos[None, :], axis=1)
                    near_v = np.where((d2 <= r) & (np.arange(len(vertices)) != v))[0]
                    for u in near_v.tolist():
                        # 下界估计（与样本一样）
                        f_hat = g_v + float(d2[u]) + h(vertices[u].pos)
                        # rewiring 也要有机会改善全局，因此同样剪枝
                        if f_hat < best_cost:
                            heapq.heappush(Qe, (f_hat, _Edge(v=v, target_is_sample=False, target_id=int(u))))

        # 若已经找到可行解，继续加密 informed set（提升质量）
        if (len(vertices) + len(samples)) >= samples_budget:
            break
        add_samples(batch_size, best_cost)

    # 输出最优解路径
    if best_goal_vid is None:
        return []
    return _reconstruct_path(vertices, best_goal_vid)


# -----------------------------
# 对外接口（保持兼容）
# -----------------------------
def plan_path(env: ObstacleEnvironment2D,
              start: np.ndarray,
              goal: np.ndarray,
              resolution: float = 2.0,
              *,
              agent_radius: float = 0.5,
              safety_margin: float = 0.8,
              max_samples: int = 6000,
              batch_size: int = 260,
              seed: int = 7) -> List[np.ndarray]:
    """
    BIT* 规划最短路径（连续空间）。

    参数建议：
    - resolution: 2~4（用作碰撞线段离散步长 & 重采样间距尺度）
    - safety_margin: 编队/跟踪更稳可设 1.0~2.0
    - max_samples: 3000~12000（越大质量越高，越慢）
    - batch_size: 200~500

    返回：世界坐标路径点 list[np.ndarray]；失败返回 []
    """
    env_size = float(env.env_size)
    lo, hi = 0.0, env_size

    start = np.asarray(start, dtype=float).reshape(2)
    goal = np.asarray(goal, dtype=float).reshape(2)

    inflated_r = float(agent_radius + safety_margin)

    path = _bitstar_plan(
        env,
        start,
        goal,
        agent_radius=inflated_r,
        bounds=(lo, hi),
        resolution=float(resolution),
        max_samples=int(max_samples),
        batch_size=int(batch_size),
        seed=int(seed),
        goal_sample_rate=0.05,
    )

    if not path:
        return []

    # 后处理：捷径平滑 + 等间距重采样
    collision_step = max(0.6, float(resolution) * 0.8)
    path = shortcut_smooth(env, path, agent_radius=inflated_r, collision_step=collision_step, iters=260, seed=0)
    path = resample_path(path, spacing=max(2.0, float(resolution) * 1.2))
    return path


if __name__ == "__main__":
    # 简单示例：命令行测试规划
    env = ObstacleEnvironment2D(env_size=300.0, num_obstacles=5, seed=42)
    start = np.array([15.0, 15.0])
    goal = np.array([285.0, 285.0])

    path = plan_path(env, start, goal, resolution=3.0, agent_radius=0.5, safety_margin=1.0,
                     max_samples=7000, batch_size=280, seed=7)
    print(f"Planned {len(path)} waypoints")
    if path:
        print("First/Last:", path[0], path[-1])


