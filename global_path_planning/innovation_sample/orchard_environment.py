"""
果园环境生成和管理模块
基于栅格占用图生成圆形障碍物
"""
import sys
import os
import math
import numpy as np
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from vehicle.vehicle_collision import CircleObstacle
from global_path_planning.innovation_sample.hybrid_sampler import GoalRectangle


@dataclass
class OrchardEnvironment:
    """果园环境数据类"""
    obstacles: list  # CircleObstacle 列表
    corridors: list  # SamplingCorridor 列表
    goal_rectangle: GoalRectangle
    start_pos: tuple  # (x, y) 起点坐标
    goal_pos: tuple   # (x, y) 目标坐标
    bounds: tuple     # (x_min, x_max, y_min, y_max)
    description: str  # 环境描述
    start_goal_pairs: list = None  # 可选：多对起点终点 [((x1,y1), (x2,y2)), ...]


def make_goal_rectangle(start_pos, goal_pos, length, width, forward_offset=0.0):
    """创建动态矩形的初始配置；规划时锚点会更新为当前末端节点。"""
    if length <= 0.0 or width <= 0.0:
        raise ValueError("矩形的 length 和 width 必须大于 0")
    return GoalRectangle(
        anchor_x=float(start_pos[0]),
        anchor_y=float(start_pos[1]),
        length=float(length),
        width=float(width),
        forward_offset=float(forward_offset),
    )


def grid_to_obstacles(occ_grid, cell_size=1.0, min_cluster_size=1, obstacle_safety_margin=0.4):
    """
    将占用栅格转换为圆形障碍物
    
    Args:
        occ_grid: 二进制占用栅格 (n x n)
        cell_size: 栅格单元大小（米）
        min_cluster_size: 最小聚类大小
    
    Returns:
        CircleObstacle 列表
    """
    obstacles = []
    visited = set()
    
    # 查找连通分量（聚类）
    for i in range(occ_grid.shape[0]):
        for j in range(occ_grid.shape[1]):
            if occ_grid[i, j] and (i, j) not in visited:
                # BFS 找聚类
                cluster = bfs_cluster(occ_grid, i, j, visited)
                
                if len(cluster) >= min_cluster_size:
                    # 计算聚类的中心和包含圆
                    center_i = np.mean([c[0] for c in cluster])
                    center_j = np.mean([c[1] for c in cluster])
                    
                    # 计算最小包含圆的半径
                    max_dist = 0
                    for ci, cj in cluster:
                        dist = np.sqrt((ci - center_i)**2 + (cj - center_j)**2)
                        max_dist = max(max_dist, dist)
                    
                    radius = (max_dist + obstacle_safety_margin) * cell_size  # 可配置的安全边距
                    
                    obs = CircleObstacle(
                        x=center_j * cell_size,
                        y=center_i * cell_size,
                        radius=radius
                    )
                    obstacles.append(obs)
    
    return obstacles


def bfs_cluster(grid, start_i, start_j, visited):
    """使用 BFS 找连通分量"""
    cluster = []
    queue = [(start_i, start_j)]
    visited.add((start_i, start_j))
    
    while queue:
        i, j = queue.pop(0)
        cluster.append((i, j))
        
        # 检查 4 邻域
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if (0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1] and
                grid[ni, nj] and (ni, nj) not in visited):
                visited.add((ni, nj))
                queue.append((ni, nj))
    
    return cluster


def make_orchard_environment(
    seed=42,
    grid_size=90,
    cell_size=1.0,
    rectangle_length=30.0,
    rectangle_width=30.0,
    rectangle_forward_offset=0.0,
    num_obstacles=18,
    obstacle_margin=5,
    obstacle_min_size=1,
    obstacle_max_size=4,
    obstacle_safety_margin=0.4,
):
    """
    生成果园仿真环境
    
    Args:
        seed: 随机种子
        grid_size: 栅格大小
        cell_size: 栅格单元对应的实际距离
    
    Returns:
        OrchardEnvironment 对象
    """
    import time
    total_start = time.time()
    
    print(f"[环境] 开始生成环境 (seed={seed}, grid_size={grid_size}x{grid_size})")
    
    rng = np.random.default_rng(seed)
    n = grid_size
    
    # 生成占用栅格
    print("[环境] 初始化栅格...")
    occ = np.zeros((n, n), dtype=bool)
    
    # 生成大量随机分散的小障碍物（类似图中的样子）
    print("[环境] 生成随机障碍物...")
    print(f"[环境] 将生成 {num_obstacles} 个障碍物...")

    for i, _ in enumerate(range(num_obstacles)):
        if i % 100 == 0:
            print(f"[环境] 已生成 {i}/{num_obstacles} 个障碍物...")
        # 随机位置
        x = rng.integers(obstacle_margin, n - obstacle_margin)
        y = rng.integers(obstacle_margin, n - obstacle_margin)

        # 随机大小
        width = rng.integers(obstacle_min_size, obstacle_max_size)
        height = rng.integers(obstacle_min_size, obstacle_max_size)
        
        # 放置障碍物
        x_start = max(0, x - width // 2)
        x_end = min(n, x + width // 2 + 1)
        y_start = max(0, y - height // 2)
        y_end = min(n, y + height // 2 + 1)
        
        occ[x_start:x_end, y_start:y_end] = 1
    
    print("[环境] 转换为圆形障碍物...")
    # 转换为圆形障碍物
    obstacles = grid_to_obstacles(occ, cell_size=cell_size, obstacle_safety_margin=obstacle_safety_margin)
    print(f"[环境] 得到 {len(obstacles)} 个圆形障碍物")
    
    # 清除起点和目标周围的障碍
    print("[环境] 清理起点和目标...")
    start = (n // 2, 5)  # 左下
    goal = (n // 2, n - 5)  # 右下
    
    obstacles = [
        obs for obs in obstacles 
        if not (obs_overlaps_point(obs, start, cell_size) or 
                obs_overlaps_point(obs, goal, cell_size))
    ]
    
    # 坐标转换：从栅格坐标 (i, j) 转换为实际坐标 (x, y)
    # 栅格坐标中 i 是行（对应 y），j 是列（对应 x）
    start_pos = (start[1] * cell_size, start[0] * cell_size)
    goal_pos = (goal[1] * cell_size, goal[0] * cell_size)
    bounds = (0, n * cell_size, 0, n * cell_size)
    
    # 清空走廊（因为这种环境不需要规则走廊）
    corridors = []
    goal_rectangle = make_goal_rectangle(
        start_pos,
        goal_pos,
        rectangle_length,
        rectangle_width,
        rectangle_forward_offset,
    )
    
    elapsed = time.time() - total_start
    print(f"[环境] ✓ 环境生成完成 ({elapsed:.2f}s, {len(obstacles)} 个有效障碍物)")
    
    return OrchardEnvironment(
        obstacles=obstacles,
        corridors=corridors,
        goal_rectangle=goal_rectangle,
        start_pos=start_pos,
        goal_pos=goal_pos,
        bounds=bounds,
        description=f"随机环境 (网格{n}x{n}, 种子{seed}, {len(obstacles)}个障碍物)"
    )


def obs_overlaps_point(obstacle, grid_point, cell_size=1.0):
    """检查障碍物是否与栅格点重叠"""
    obs_x, obs_y = obstacle.x, obstacle.y
    point_x = grid_point[1] * cell_size
    point_y = grid_point[0] * cell_size
    dist = np.sqrt((obs_x - point_x)**2 + (obs_y - point_y)**2)
    return dist < obstacle.radius + 1.0


def make_complex_environment(
    seed=42,
    grid_size=90,
    cell_size=1.0,
    rectangle_length=30.0,
    rectangle_width=30.0,
    rectangle_forward_offset=0.0,
    obstacle_safety_margin=0.4,
):
    """生成含更多、更大障碍物的复杂果园环境。"""
    rng = np.random.default_rng(seed)
    n = grid_size
    occ = np.zeros((n, n), dtype=bool)

    for _ in range(35):
        row = rng.integers(8, n - 8)
        col = rng.integers(8, n - 8)
        height = rng.integers(2, 7)
        width = rng.integers(2, 7)
        row_start = max(0, row - height // 2)
        row_end = min(n, row + height // 2 + 1)
        col_start = max(0, col - width // 2)
        col_end = min(n, col + width // 2 + 1)
        occ[row_start:row_end, col_start:col_end] = True

    obstacles = grid_to_obstacles(occ, cell_size=cell_size, obstacle_safety_margin=obstacle_safety_margin)
    start = (10, 10)
    goal = (n - 10, n - 10)
    obstacles = [
        obstacle
        for obstacle in obstacles
        if not obs_overlaps_point(obstacle, start, cell_size)
        and not obs_overlaps_point(obstacle, goal, cell_size)
    ]

    start_pos = (start[1] * cell_size, start[0] * cell_size)
    goal_pos = (goal[1] * cell_size, goal[0] * cell_size)
    bounds = (0, n * cell_size, 0, n * cell_size)
    goal_rectangle = make_goal_rectangle(
        start_pos,
        goal_pos,
        rectangle_length,
        rectangle_width,
        rectangle_forward_offset,
    )

    return OrchardEnvironment(
        obstacles=obstacles,
        corridors=[],
        goal_rectangle=goal_rectangle,
        start_pos=start_pos,
        goal_pos=goal_pos,
        bounds=bounds,
        description=(
            f"复杂随机环境 (网格{n}x{n}, 种子{seed}, "
            f"{len(obstacles)}个障碍物)"
        ),
    )


def make_hybrid_benchmark_environment(
    scenario="single_blocker",
    rectangle_length=30.0,
    rectangle_width=20.0,
):
    """构造用于比较 GoalBias 与 Hybrid 的确定性圆障碍果园场景。"""
    bounds = (0.0, 90.0, 0.0, 90.0)
    scenarios = {
        # 目标直连线被单棵大树截断，突出一次切向子目标的作用。
        "single_blocker": {
            "start": (8.0, 45.0),
            "goal": (82.0, 45.0),
            "obstacles": [
                (43.0, 45.0, 5.0),
                (28.0, 27.0, 3.0), (28.0, 65.0, 3.0),
                (60.0, 28.0, 3.5), (61.0, 64.0, 3.5),
            ],
            "description": "Hybrid基准1：单圆封锁目标直线",
        },
        # 多棵树交错压住中心线，但上下均留有可行空间。
        "staggered_trees": {
            "start": (8.0, 45.0),
            "goal": (82.0, 45.0),
            "obstacles": [
                (27.0, 42.0, 3.2),
                (40.0, 49.0, 3.2),
                (53.0, 41.0, 3.2),
                (66.0, 49.0, 3.2),
                (27.0, 68.0, 3.0), (40.0, 22.0, 3.0),
                (54.0, 68.0, 3.0), (67.0, 22.0, 3.0),
            ],
            "description": "Hybrid基准2：交错圆形果树连续绕行",
        },
        # 两排果树形成窄而清晰的行间通道，矩形采样应减少无效侧向探索。
        "row_corridor": {
            "start": (8.0, 45.0),
            "goal": (82.0, 45.0),
            "obstacles": [
                *[(x, 34.0, 3.0) for x in range(18, 79, 10)],
                *[(x, 56.0, 3.0) for x in range(18, 79, 10)],
                (44.0, 45.0, 2.8),
            ],
            "description": "Hybrid基准3：果树行间窄通道与局部阻挡",
        },
    }
    if scenario not in scenarios:
        raise ValueError(
            f"未知Hybrid基准场景: {scenario}; 可选: {', '.join(scenarios)}"
        )

    config = scenarios[scenario]
    start_pos = config["start"]
    goal_pos = config["goal"]
    obstacles = [CircleObstacle(x, y, radius) for x, y, radius in config["obstacles"]]
    return OrchardEnvironment(
        obstacles=obstacles,
        corridors=[],
        goal_rectangle=make_goal_rectangle(
            start_pos, goal_pos, rectangle_length, rectangle_width
        ),
        start_pos=start_pos,
        goal_pos=goal_pos,
        bounds=bounds,
        description=config["description"],
    )


def make_density_environment(
    obstacle_count=20,
    seed=0,
    rectangle_length=30.0,
    rectangle_width=20.0,
    obstacle_clearance=2.0,
):
    """生成固定范围、彼此不重叠的零散圆形障碍地图。

    ``obstacle_clearance`` 是两个障碍物边缘之间保留的最小空隙。
    """
    rng = np.random.default_rng(seed)
    bounds = (0.0, 90.0, 0.0, 90.0)
    start_pos = (8.0, 45.0)
    goal_pos = (82.0, 45.0)
    obstacles = []
    attempts = 0
    max_attempts = max(1000, int(obstacle_count) * 500)
    clearance = max(0.0, float(obstacle_clearance))
    while len(obstacles) < int(obstacle_count) and attempts < max_attempts:
        attempts += 1
        x = float(rng.uniform(5.0, 85.0))
        y = float(rng.uniform(5.0, 85.0))
        radius = float(rng.uniform(1.8, 3.8))
        if math.hypot(x - start_pos[0], y - start_pos[1]) < radius + 6.0:
            continue
        if math.hypot(x - goal_pos[0], y - goal_pos[1]) < radius + 6.0:
            continue
        if any(
            math.hypot(x - obstacle.x, y - obstacle.y)
            < radius + obstacle.radius + clearance
            for obstacle in obstacles
        ):
            continue
        obstacles.append(CircleObstacle(x, y, radius))

    if len(obstacles) < int(obstacle_count):
        raise RuntimeError(
            f"无法在 {max_attempts} 次尝试内放置 {obstacle_count} 个互不重叠的障碍物；"
            f"当前仅放置 {len(obstacles)} 个，请减少数量或 obstacle_clearance。"
        )

    return OrchardEnvironment(
        obstacles=obstacles,
        corridors=[],
        goal_rectangle=make_goal_rectangle(
            start_pos, goal_pos, rectangle_length, rectangle_width
        ),
        start_pos=start_pos,
        goal_pos=goal_pos,
        bounds=bounds,
        description=(
            f"密度实验地图（{obstacle_count}个互不重叠圆形障碍，"
            f"边缘间隔≥{clearance:.1f}m，地图种子{seed}）"
        ),
    )


def make_overlap_environment(
    overlap_percent=0,
    seed=0,
    obstacle_count=30,
    rectangle_length=30.0,
    rectangle_width=20.0,
):
    """生成固定障碍数量、不同圆形障碍重叠程度的随机地图。"""
    rng = np.random.default_rng(seed)
    bounds = (0.0, 90.0, 0.0, 90.0)
    start_pos, goal_pos = (8.0, 45.0), (82.0, 45.0)
    obstacles = []
    overlap_probability = max(0.0, min(1.0, overlap_percent / 100.0))
    attempts = 0
    while len(obstacles) < obstacle_count and attempts < obstacle_count * 200:
        attempts += 1
        radius = float(rng.uniform(1.8, 3.8))
        if obstacles and rng.random() < overlap_probability:
            parent = obstacles[int(rng.integers(0, len(obstacles)))]
            angle = float(rng.uniform(-math.pi, math.pi))
            distance = float(rng.uniform(0.25, 0.85)) * (parent.radius + radius)
            x = parent.x + distance * math.cos(angle)
            y = parent.y + distance * math.sin(angle)
        else:
            x = float(rng.uniform(5.0, 85.0))
            y = float(rng.uniform(5.0, 85.0))
        if not (3.0 <= x <= 87.0 and 3.0 <= y <= 87.0):
            continue
        if math.hypot(x - start_pos[0], y - start_pos[1]) < radius + 6.0:
            continue
        if math.hypot(x - goal_pos[0], y - goal_pos[1]) < radius + 6.0:
            continue
        obstacles.append(CircleObstacle(x, y, radius))
    return OrchardEnvironment(
        obstacles=obstacles,
        corridors=[],
        goal_rectangle=make_goal_rectangle(
            start_pos, goal_pos, rectangle_length, rectangle_width
        ),
        start_pos=start_pos,
        goal_pos=goal_pos,
        bounds=bounds,
        description=f"重叠率实验地图（{overlap_percent}%、{obstacle_count}个障碍、种子{seed}）",
    )


def make_gap_environment(
    gap_width=4.5,
    seed=0,
    rectangle_length=30.0,
    rectangle_width=20.0,
):
    """生成由圆形树冠构成的单一必经窄通道。

    ``gap_width`` 是两侧树冠边界之间的净间隙，而非圆心距。通过改变
    墙体位置和通道中心获得不同地图，但每张地图保持相同的净通行宽度。
    """
    if gap_width <= 0.0:
        raise ValueError("gap_width 必须大于 0")

    rng = np.random.default_rng(seed)
    bounds = (0.0, 90.0, 0.0, 90.0)
    start_pos, goal_pos = (8.0, 45.0), (82.0, 45.0)
    radius = float(rng.uniform(3.0, 3.4))
    wall_x = float(45.0 + rng.uniform(-2.0, 2.0))
    gate_y = float(45.0 + rng.uniform(-1.5, 1.5))
    spacing = 2.0 * radius - 0.45  # 相邻树冠轻微重叠，避免从墙体侧面穿过。

    lower_gate_center = gate_y - (radius + gap_width / 2.0)
    upper_gate_center = gate_y + (radius + gap_width / 2.0)
    wall_centers = [lower_gate_center, upper_gate_center]

    y = lower_gate_center - spacing
    while y > radius - 0.2:
        wall_centers.append(y)
        y -= spacing
    y = upper_gate_center + spacing
    while y < 90.0 - radius + 0.2:
        wall_centers.append(y)
        y += spacing

    obstacles = [CircleObstacle(wall_x, y, radius) for y in wall_centers]

    # 少量远离门洞的果树避免环境退化成完全规则的人工墙体。
    for side_x in (25.0, 66.0):
        for _ in range(2):
            x = float(side_x + rng.uniform(-4.0, 4.0))
            y = float(rng.choice((rng.uniform(10.0, 25.0), rng.uniform(65.0, 80.0))))
            r = float(rng.uniform(1.8, 2.8))
            obstacles.append(CircleObstacle(x, y, r))

    return OrchardEnvironment(
        obstacles=obstacles,
        corridors=[],
        goal_rectangle=make_goal_rectangle(
            start_pos, goal_pos, rectangle_length, rectangle_width
        ),
        start_pos=start_pos,
        goal_pos=goal_pos,
        bounds=bounds,
        description=(
            f"窄通道实验地图（净间隙{gap_width:.1f}m、地图种子{seed}）"
        ),
    )


def _corridor_values(corridor):
    if isinstance(corridor, dict):
        return [
            corridor["x1"], corridor["y1"],
            corridor["x2"], corridor["y2"], corridor["width"],
        ]
    return [
        corridor.x1, corridor.y1,
        corridor.x2, corridor.y2, corridor.width,
    ]


def save_environment(environment, npz_path):
    """把果园环境保存为可由实验脚本直接读取的NPZ文件。"""
    npz_path = os.path.abspath(npz_path)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)

    obstacles = np.asarray(
        [[obs.x, obs.y, obs.radius] for obs in environment.obstacles],
        dtype=float,
    ).reshape(-1, 3)
    corridors = np.asarray(
        [_corridor_values(corridor) for corridor in environment.corridors],
        dtype=float,
    ).reshape(-1, 5)
    rectangle = environment.goal_rectangle

    np.savez_compressed(
        npz_path,
        format_version=np.asarray([1], dtype=np.int32),
        obstacles=obstacles,
        corridors=corridors,
        start_pos=np.asarray(environment.start_pos, dtype=float),
        goal_pos=np.asarray(environment.goal_pos, dtype=float),
        bounds=np.asarray(environment.bounds, dtype=float),
        rectangle=np.asarray([
            rectangle.length,
            rectangle.width,
            rectangle.forward_offset,
        ], dtype=float),
        description=np.asarray(environment.description),
    )
    print(f"[环境] NPZ已保存: {npz_path}")
    return npz_path


def load_environment(npz_path):
    """从NPZ恢复完整果园环境。"""
    npz_path = os.path.abspath(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        required = {
            "obstacles", "corridors", "start_pos", "goal_pos",
            "bounds", "rectangle", "description",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"环境NPZ缺少字段: {', '.join(sorted(missing))}"
            )

        start_pos = tuple(float(v) for v in data["start_pos"])
        goal_pos = tuple(float(v) for v in data["goal_pos"])
        bounds = tuple(float(v) for v in data["bounds"])
        rectangle_values = data["rectangle"].astype(float)
        obstacles = [
            CircleObstacle(float(x), float(y), float(radius))
            for x, y, radius in data["obstacles"]
        ]
        corridors = [
            {
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "width": float(width),
            }
            for x1, y1, x2, y2, width in data["corridors"]
        ]
        description = str(data["description"].item())

        # 检查是否有新格式的多对起点终点（format_version >= 2）
        start_goal_pairs = None
        if "start_goal_pairs" in data.files:
            pairs_array = data["start_goal_pairs"]
            start_goal_pairs = [
                ((float(start_x), float(start_y)), (float(goal_x), float(goal_y)))
                for start_x, start_y, goal_x, goal_y in pairs_array
            ]

    environment = OrchardEnvironment(
        obstacles=obstacles,
        corridors=corridors,
        goal_rectangle=make_goal_rectangle(
            start_pos,
            goal_pos,
            length=float(rectangle_values[0]),
            width=float(rectangle_values[1]),
            forward_offset=float(rectangle_values[2]),
        ),
        start_pos=start_pos,
        goal_pos=goal_pos,
        bounds=bounds,
        description=description,
        start_goal_pairs=start_goal_pairs,  # 新增：多对起点终点
    )
    if start_goal_pairs:
        print(f"[环境] 已读取NPZ: {npz_path} (包含{len(start_goal_pairs)}对起点终点)")
    else:
        print(f"[环境] 已读取NPZ: {npz_path}")
    return environment


def plot_environment(environment, image_path=None, show=True):
    """显示果园环境，并可同时保存PNG图片。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Circle

    available_fonts = {
        font.name for font in font_manager.fontManager.ttflist
    }
    for font_name in ("PingFang SC", "Heiti SC", "SimHei", "Noto Sans CJK SC"):
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 9))
    for obstacle in environment.obstacles:
        ax.add_patch(Circle(
            (obstacle.x, obstacle.y),
            obstacle.radius,
            facecolor="lightcoral",
            edgecolor="red",
            alpha=0.65,
        ))

    ax.plot(*environment.start_pos, "go", markersize=12, label="Start")
    ax.plot(*environment.goal_pos, "r*", markersize=20, label="Goal")
    x_min, x_max, y_min, y_max = environment.bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Orchard environment | obstacles={len(environment.obstacles)}"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()

    if image_path:
        image_path = os.path.abspath(image_path)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        fig.savefig(image_path, dpi=180, bbox_inches="tight")
        print(f"[环境] 图片已保存: {image_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == "__main__":
    output_dir = os.path.dirname(__file__)
    env = make_complex_environment(
        seed=12,
        grid_size=90,
        cell_size=0.5,
        rectangle_length=15.0,
        rectangle_width=30.0,
    )
    save_environment(
        env,
        os.path.join(output_dir, "orchard_environment.npz"),
    )
    plot_environment(
        env,
        image_path=os.path.join(output_dir, "orchard_environment.png"),
        show=True,
    )
    print(f"环境: {env.description}")
    print(f"起点: {env.start_pos}")
    print(f"目标: {env.goal_pos}")
    print(f"障碍物数: {len(env.obstacles)}")
    print(f"走廊数: {len(env.corridors)}")
