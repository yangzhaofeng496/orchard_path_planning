"""
消融实验：对比 RRT* 的三种配置
1. 基础 RRT* (无目标偏置，无切向圆)
2. RRT* + 目标偏置
3. RRT* + 目标偏置 + 切向圆

每组配置测试100次，统计关键指标并生成SVG表格
支持多场景并行运行
"""
import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm
# from multiprocessing import Pool, Manager, cpu_count
# import multiprocessing

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from vehicle.reeds_shepp_path import Pose
from vehicle.vehicle_collision import VehicleGeometry, CircleObstacle
from global_path_planning.innovation_sample.ackermann_rrt_star import AckermannRRTStar
from global_path_planning.innovation_sample.orchard_environment import load_environment


def generate_random_pose(bounds, obstacles, vehicle, rng, min_clearance=3.0, max_attempts=1000):
    """
    在地图中随机生成一个不与障碍物碰撞的位置

    Args:
        bounds: (x_min, x_max, y_min, y_max)
        obstacles: 障碍物列表
        vehicle: 车辆几何参数
        rng: numpy随机数生成器
        min_clearance: 与障碍物的最小安全距离
        max_attempts: 最大尝试次数

    Returns:
        Pose: 随机生成的位姿
    """
    x_min, x_max, y_min, y_max = bounds

    # 留出边界余量
    margin = 5.0
    x_min += margin
    x_max -= margin
    y_min += margin
    y_max -= margin

    for attempt in range(max_attempts):
        # 随机生成位置
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        yaw = rng.uniform(-np.pi, np.pi)

        # 检查与所有障碍物的距离
        is_valid = True
        for obs in obstacles:
            dist = np.hypot(x - obs.x, y - obs.y)
            # 考虑车辆尺寸和安全间隙
            required_clearance = obs.radius + vehicle.width / 2 + vehicle.safety_margin + min_clearance
            if dist < required_clearance:
                is_valid = False
                break

        if is_valid:
            return Pose(x=x, y=y, yaw=yaw)

    raise RuntimeError(f"无法在 {max_attempts} 次尝试内找到有效的随机位姿")


def generate_random_start_goal(env, vehicle, seed=None, min_distance=30.0):
    """
    为环境生成随机的起点和终点

    Args:
        env: OrchardEnvironment 对象
        vehicle: 车辆几何参数
        seed: 随机种子
        min_distance: 起点和终点之间的最小距离

    Returns:
        tuple: (start_pose, goal_pose)
    """
    rng = np.random.default_rng(seed)

    # 生成起点
    start_pose = generate_random_pose(env.bounds, env.obstacles, vehicle, rng)

    # 生成终点，确保与起点有足够距离
    max_attempts = 1000
    for attempt in range(max_attempts):
        goal_pose = generate_random_pose(env.bounds, env.obstacles, vehicle, rng)

        # 检查起点和终点的距离
        distance = np.hypot(goal_pose.x - start_pose.x, goal_pose.y - start_pose.y)
        if distance >= min_distance:
            return start_pose, goal_pose

    raise RuntimeError(f"无法在 {max_attempts} 次尝试内找到满足最小距离的起点和终点")


def load_or_generate_poses(map_path, vehicle, use_map_poses=False, seed=None, pair_index=0):
    """
    加载地图并获取起点终点（从地图文件或随机生成）

    Args:
        map_path: 地图文件路径
        vehicle: 车辆几何参数
        use_map_poses: 是否使用地图文件中的起点终点
        seed: 随机种子（当use_map_poses=False时使用）
        pair_index: 使用地图中第几对起点终点（默认0）

    Returns:
        tuple: (env, start_pose, goal_pose)
    """
    env = load_environment(map_path)

    if use_map_poses:
        # 使用地图文件中的起点终点
        if env.start_goal_pairs and len(env.start_goal_pairs) > pair_index:
            # 使用指定索引的起点终点对
            start, goal = env.start_goal_pairs[pair_index]
            start_pose = Pose(x=start[0], y=start[1], yaw=0.0)
            goal_pose = Pose(x=goal[0], y=goal[1], yaw=0.0)
        else:
            # 回退到默认起点终点（向后兼容）
            start_pose = Pose(x=env.start_pos[0], y=env.start_pos[1], yaw=0.0)
            goal_pose = Pose(x=env.goal_pos[0], y=env.goal_pos[1], yaw=0.0)
    else:
        # 随机生成起点终点
        start_pose, goal_pose = generate_random_start_goal(env, vehicle, seed=seed)

    return env, start_pose, goal_pose


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    goal_probability: float
    tangent_probability: float
    use_tangent_guidance: bool


@dataclass
class TrialResult:
    """单次实验结果"""
    success: bool
    path_length: float
    planning_time: float
    num_nodes: int
    num_iterations: int

    @staticmethod
    def failed():
        """创建失败结果"""
        return TrialResult(
            success=False,
            path_length=float('inf'),
            planning_time=0.0,
            num_nodes=0,
            num_iterations=0
        )


class AblationExperiment:
    """消融实验类"""

    def __init__(
        self,
        map_path: str,
        start_pose: Pose,
        goal_pose: Pose,
        vehicle: VehicleGeometry,
        output_dir: str = "./results",
        env: Optional['OrchardEnvironment'] = None
    ):
        self.map_path = map_path
        self.start_pose = start_pose
        self.goal_pose = goal_pose
        self.vehicle = vehicle
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载环境（如果未提供）
        if env is None:
            print(f"加载地图: {map_path}")
            self.env = load_environment(map_path)
            print(f"障碍物数量: {len(self.env.obstacles)}")
        else:
            self.env = env

        # 定义三种配置（参考 interactive_ablation_planner.py）
        self.configs = [
            ExperimentConfig(
                name="Baseline",
                goal_probability=0.0,
                tangent_probability=0.0,
                use_tangent_guidance=False,
            ),
            ExperimentConfig(
                name="GoalBias",
                goal_probability=0.20,  # 修改为0.20，与interactive保持一致
                tangent_probability=0.0,
                use_tangent_guidance=False,
            ),
            ExperimentConfig(
                name="GoalBias+Tangent",
                goal_probability=0.20,  # 与interactive保持一致
                tangent_probability=0.10,  # 修改为0.10，与interactive保持一致
                use_tangent_guidance=True,
            ),
        ]

    def run_single_trial(
        self,
        config: ExperimentConfig,
        max_iterations: int = 2500,
        random_seed: int = None,
    ) -> TrialResult:
        """运行单次实验"""
        try:
            # 根据配置决定是否启用切向连接器（参考 interactive_ablation_planner.py）
            use_hybrid_sampling = (
                config.goal_probability > 0.0
                or config.tangent_probability > 0.0
            )
            enable_tangent_connectors = config.use_tangent_guidance

            # 创建规划器（参数与 interactivate_exp.py 保持一致）
            planner = AckermannRRTStar(
                start=self.start_pose,
                goal=self.goal_pose,
                bounds=self.env.bounds,
                vehicle=self.vehicle,
                obstacles=self.env.obstacles,
                curvature=1.0 / 3.0,  # 最小转弯半径 3m
                use_ackermann_constraints=False,  # 所有消融组统一使用经过碰撞检测的直线段连接
                expand_length=3.0,
                step_size=0.08,
                max_iterations=max_iterations,
                near_radius=5.0,
                use_hybrid_sampling=use_hybrid_sampling,
                goal_probability=config.goal_probability,
                tangent_probability=config.tangent_probability,
                adaptive_sampling_probabilities=False,  # 与interactivate_exp保持一致
                corridor_probability=0.0,
                rectangle_probability=0.45,  # 修改为0.45，与interactivate_exp保持一致
                allow_reverse=enable_tangent_connectors,
                use_tangent_guidance=config.use_tangent_guidance,
                shrink_probability=0.35,  # 新增：与interactivate_exp保持一致
                shrink_length_factor=0.70,  # 新增：与interactivate_exp保持一致
                shrink_width_factor=0.70,  # 新增：与interactivate_exp保持一致
                shrink_activation_distance=18.0,  # 新增：与interactivate_exp保持一致
                near_anchor_probability=0.55,  # 新增：与interactivate_exp保持一致
                near_anchor_length_ratio=0.40,  # 新增：与interactivate_exp保持一致
                cluster_shape="ellipse",  # 新增：与interactivate_exp保持一致
                rectangle_anchor_mode="closest_to_goal",  # 新增：与interactivate_exp保持一致
                use_goal_connector=enable_tangent_connectors,  # 根据配置动态设置
                relax_goal_yaw=False,
                random_seed=random_seed if random_seed is not None else int(time.time() * 1000) % 100000,
            )

            # 纯批处理规划，不创建窗口或注册绘图回调。
            start_time = time.time()
            result = planner.planning()

            planning_time = time.time() - start_time

            if result is None:
                return TrialResult.failed()

            # 计算路径长度
            path_x, path_y, _, _ = result
            path_length = 0.0
            for i in range(1, len(path_x)):
                path_length += np.hypot(path_x[i] - path_x[i-1], path_y[i] - path_y[i-1])

            return TrialResult(
                success=True,
                path_length=path_length,
                planning_time=planning_time,
                num_nodes=len(planner.nodes),
                num_iterations=planner.first_solution_iteration if planner.first_solution_iteration else max_iterations
            )

        except Exception as e:
            print(f"  错误: {e}")
            return TrialResult.failed()

    def run_experiment(self, num_trials: int = 100) -> pd.DataFrame:
        """
        运行完整实验

        Args:
            num_trials: 每个配置的测试次数
        """
        print("\n" + "="*70)
        print(f"开始消融实验")
        print(f"起点: ({self.start_pose.x:.2f}, {self.start_pose.y:.2f}, {np.degrees(self.start_pose.yaw):.1f}°)")
        print(f"终点: ({self.goal_pose.x:.2f}, {self.goal_pose.y:.2f}, {np.degrees(self.goal_pose.yaw):.1f}°)")
        print(f"每组配置测试: {num_trials} 次")
        print("="*70 + "\n")

        all_results = []

        for config in self.configs:
            print(f"\n测试配置: {config.name}")
            print(f"  目标偏置概率: {config.goal_probability:.2f}")
            print(f"  切向圆概率: {config.tangent_probability:.2f}")
            print(f"  切向引导: {config.use_tangent_guidance}")
            print("-" * 70)

            for trial_idx in tqdm(range(num_trials), desc=f"  {config.name}", ncols=100):
                result = self.run_single_trial(
                    config,
                    random_seed=42 + trial_idx,
                )

                all_results.append({
                    'config': config.name,
                    'trial': trial_idx + 1,
                    'success': result.success,
                    'path_length': result.path_length if result.success else np.nan,
                    'planning_time': result.planning_time if result.success else np.nan,
                    'num_nodes': result.num_nodes if result.success else np.nan,
                    'num_iterations': result.num_iterations if result.success else np.nan,
                })

        df = pd.DataFrame(all_results)
        return df

    def compute_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算统计指标"""
        stats = []

        for config_name in df['config'].unique():
            config_df = df[df['config'] == config_name]
            success_df = config_df[config_df['success']]

            num_trials = len(config_df)
            num_success = len(success_df)
            success_rate = num_success / num_trials * 100

            stats.append({
                '配置': config_name,
                '成功次数': num_success,
                '成功率(%)': f"{success_rate:.1f}",
                '平均路径长度(m)': f"{success_df['path_length'].mean():.2f}" if num_success > 0 else "N/A",
                '路径长度std': f"{success_df['path_length'].std():.2f}" if num_success > 0 else "N/A",
                '平均规划时间(s)': f"{success_df['planning_time'].mean():.3f}" if num_success > 0 else "N/A",
                '规划时间std': f"{success_df['planning_time'].std():.3f}" if num_success > 0 else "N/A",
                '平均节点数': f"{success_df['num_nodes'].mean():.0f}" if num_success > 0 else "N/A",
                '平均迭代数': f"{success_df['num_iterations'].mean():.0f}" if num_success > 0 else "N/A",
            })

        return pd.DataFrame(stats)

def process_single_map(args):
    """
    处理单个地图的函数（用于多进程）

    Args:
        args: 包含 (map_path, map_idx, total_maps, vehicle, base_output_dir, use_random_poses, map_seed)

    Returns:
        tuple: (map_name, stats_df_list, success_flag)
    """
    map_path, map_idx, total_maps, vehicle, base_output_dir, use_random_poses, map_seed = args
    map_name = os.path.basename(map_path).replace('.npz', '')

    print(f"\n{'='*70}")
    print(f"[进程 {os.getpid()}] [{map_idx}/{total_maps}] 开始测试地图: {map_name}")
    print(f"{'='*70}")

    # 为每个地图创建独立输出目录
    output_dir = os.path.join(base_output_dir, map_name)

    try:
        # 先加载地图检查有多少对起点终点
        env = load_environment(map_path)

        num_pairs = len(env.start_goal_pairs) if env.start_goal_pairs else 1
        print(f"[{map_name}] 检测到 {num_pairs} 对起点终点")

        all_stats = []
        all_pose_info = []

        # 遍历每对起点终点
        for pair_idx in range(num_pairs):
            print(f"\n[{map_name}] --- 测试起点终点对 {pair_idx + 1}/{num_pairs} ---")

            # 加载指定的起点终点对
            env, start_pose, goal_pose = load_or_generate_poses(
                map_path,
                vehicle,
                use_map_poses=not use_random_poses,
                seed=map_seed + pair_idx,
                pair_index=pair_idx
            )

            print(f"[{map_name}] 起点: ({start_pose.x:.2f}, {start_pose.y:.2f}, {np.degrees(start_pose.yaw):.1f}°)")
            print(f"[{map_name}] 终点: ({goal_pose.x:.2f}, {goal_pose.y:.2f}, {np.degrees(goal_pose.yaw):.1f}°)")

            # 为每对起点终点创建子目录
            pair_output_dir = os.path.join(output_dir, f"pair_{pair_idx}")

            # 创建实验
            experiment = AblationExperiment(
                map_path=map_path,
                start_pose=start_pose,
                goal_pose=goal_pose,
                vehicle=vehicle,
                output_dir=pair_output_dir,
                env=env
            )

            results_df = experiment.run_experiment(num_trials=50)

            # 保存原始数据
            results_csv_path = experiment.output_dir / "raw_results.csv"
            results_df.to_csv(results_csv_path, index=False, encoding='utf-8')
            print(f"[{map_name}] pair_{pair_idx} 原始数据已保存: {results_csv_path}")

            # 计算统计指标
            stats_df = experiment.compute_statistics(results_df)

            # 添加地图名称列和pair索引
            stats_df.insert(0, '地图', map_name)
            stats_df.insert(1, 'pair索引', pair_idx)

            # 保存起终点信息
            pose_info = pd.DataFrame([{
                '地图': map_name,
                'pair索引': pair_idx,
                '起点X': start_pose.x,
                '起点Y': start_pose.y,
                '起点Yaw': np.degrees(start_pose.yaw),
                '终点X': goal_pose.x,
                '终点Y': goal_pose.y,
                '终点Yaw': np.degrees(goal_pose.yaw),
                '直线距离': np.hypot(goal_pose.x - start_pose.x, goal_pose.y - start_pose.y)
            }])
            pose_csv_path = experiment.output_dir / "start_goal_poses.csv"
            pose_info.to_csv(pose_csv_path, index=False, encoding='utf-8')

            # 打印统计结果
            print(f"[{map_name}] pair_{pair_idx} 统计结果:")
            print(stats_df.to_string(index=False))

            # 保存统计数据
            stats_csv_path = experiment.output_dir / "statistics.csv"
            stats_df.to_csv(stats_csv_path, index=False, encoding='utf-8')

            all_stats.append(stats_df)
            all_pose_info.append(pose_info)

        # 保存所有pair的汇总信息
        if all_stats:
            combined_stats = pd.concat(all_stats, ignore_index=True)
            combined_poses = pd.concat(all_pose_info, ignore_index=True)

            summary_csv_path = Path(output_dir) / "all_pairs_summary.csv"
            combined_stats.to_csv(summary_csv_path, index=False, encoding='utf-8')

            poses_csv_path = Path(output_dir) / "all_pairs_poses.csv"
            combined_poses.to_csv(poses_csv_path, index=False, encoding='utf-8')

            print(f"\n[{map_name}] ✅ 所有 {num_pairs} 对测试完成")
            print(f"[{map_name}] 汇总数据已保存: {summary_csv_path}")

        return (map_name, all_stats, True)

    except Exception as e:
        print(f"\n[{map_name}] ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return (map_name, None, False)


def run_sequential_experiments(scene_configs, use_random_poses=True, random_seed=42):
    """
    顺序运行多个场景的实验（取消多进程）

    Args:
        scene_configs: 场景配置列表，每个配置包含 (scene_name, map_dir)
        use_random_poses: 是否随机生成起点终点（False则使用地图文件中的）
        random_seed: 随机种子基数
    """
    print(f"\n{'='*70}")
    print(f"顺序实验配置")
    print(f"{'='*70}")
    print(f"场景数量: {len(scene_configs)}")
    print(f"执行模式: 顺序执行（非多进程）")
    print(f"随机生成起终点: {use_random_poses}")
    if use_random_poses:
        print(f"随机种子: {random_seed}")
    print(f"{'='*70}\n")

    # 车辆参数（所有场景共用）
    vehicle = VehicleGeometry(
        front_length=1.25,
        rear_length=0.35,
        width=0.8,
        safety_margin=0.18
    )

    all_scene_results = {}

    for scene_idx, (scene_name, map_dir) in enumerate(scene_configs, 1):
        print(f"\n{'#'*70}")
        print(f"场景 [{scene_idx}/{len(scene_configs)}]: {scene_name}")
        print(f"地图目录: {map_dir}")
        print(f"{'#'*70}")

        # 搜索地图文件
        import glob
        map_files = sorted(glob.glob(os.path.join(map_dir, "*.npz")))

        if not map_files:
            print(f"⚠️  警告: 在 {map_dir} 中未找到.npz文件，跳过此场景")
            continue

        print(f"找到 {len(map_files)} 个地图文件")

        # 准备输出目录
        base_output_dir = f"./results/{scene_name}"

        # 顺序处理每个地图
        scene_results = []
        success_count = 0

        for map_idx, map_path in enumerate(map_files, 1):
            map_name = os.path.basename(map_path).replace('.npz', '')
            map_seed = random_seed + map_idx

            print(f"\n{'='*70}")
            print(f"[{map_idx}/{len(map_files)}] 开始测试地图: {map_name}")
            print(f"{'='*70}")

            # 为每个地图创建独立输出目录
            output_dir = os.path.join(base_output_dir, map_name)

            try:
                # 先加载地图检查有多少对起点终点
                env = load_environment(map_path)

                num_pairs = len(env.start_goal_pairs) if env.start_goal_pairs else 1
                print(f"[{map_name}] 检测到 {num_pairs} 对起点终点")

                all_stats = []
                all_pose_info = []

                # 遍历每对起点终点
                for pair_idx in range(num_pairs):
                    print(f"\n[{map_name}] --- 测试起点终点对 {pair_idx + 1}/{num_pairs} ---")

                    # 加载指定的起点终点对
                    env, start_pose, goal_pose = load_or_generate_poses(
                        map_path,
                        vehicle,
                        use_map_poses=not use_random_poses,
                        seed=map_seed + pair_idx,
                        pair_index=pair_idx
                    )

                    print(f"[{map_name}] 起点: ({start_pose.x:.2f}, {start_pose.y:.2f}, {np.degrees(start_pose.yaw):.1f}°)")
                    print(f"[{map_name}] 终点: ({goal_pose.x:.2f}, {goal_pose.y:.2f}, {np.degrees(goal_pose.yaw):.1f}°)")

                    # 为每对起点终点创建子目录
                    pair_output_dir = os.path.join(output_dir, f"pair_{pair_idx}")

                    # 创建实验
                    experiment = AblationExperiment(
                        map_path=map_path,
                        start_pose=start_pose,
                        goal_pose=goal_pose,
                        vehicle=vehicle,
                        output_dir=pair_output_dir,
                        env=env
                    )

                    # 运行实验
                    results_df = experiment.run_experiment(num_trials=50)

                    # 保存原始数据
                    results_csv_path = experiment.output_dir / "raw_results.csv"
                    results_df.to_csv(results_csv_path, index=False, encoding='utf-8')
                    print(f"[{map_name}] pair_{pair_idx} 原始数据已保存: {results_csv_path}")

                    # 计算统计指标
                    stats_df = experiment.compute_statistics(results_df)

                    # 添加地图名称列和pair索引
                    stats_df.insert(0, '地图', map_name)
                    stats_df.insert(1, 'pair索引', pair_idx)

                    # 保存起终点信息
                    pose_info = pd.DataFrame([{
                        '地图': map_name,
                        'pair索引': pair_idx,
                        '起点X': start_pose.x,
                        '起点Y': start_pose.y,
                        '起点Yaw': np.degrees(start_pose.yaw),
                        '终点X': goal_pose.x,
                        '终点Y': goal_pose.y,
                        '终点Yaw': np.degrees(goal_pose.yaw),
                        '直线距离': np.hypot(goal_pose.x - start_pose.x, goal_pose.y - start_pose.y)
                    }])
                    pose_csv_path = experiment.output_dir / "start_goal_poses.csv"
                    pose_info.to_csv(pose_csv_path, index=False, encoding='utf-8')

                    # 打印统计结果
                    print(f"[{map_name}] pair_{pair_idx} 统计结果:")
                    print(stats_df.to_string(index=False))

                    # 保存统计数据
                    stats_csv_path = experiment.output_dir / "statistics.csv"
                    stats_df.to_csv(stats_csv_path, index=False, encoding='utf-8')

                    all_stats.append(stats_df)
                    all_pose_info.append(pose_info)

                # 保存所有pair的汇总信息
                if all_stats:
                    combined_stats = pd.concat(all_stats, ignore_index=True)
                    combined_poses = pd.concat(all_pose_info, ignore_index=True)

                    summary_csv_path = Path(output_dir) / "all_pairs_summary.csv"
                    combined_stats.to_csv(summary_csv_path, index=False, encoding='utf-8')

                    poses_csv_path = Path(output_dir) / "all_pairs_poses.csv"
                    combined_poses.to_csv(poses_csv_path, index=False, encoding='utf-8')

                    print(f"\n[{map_name}] ✅ 所有 {num_pairs} 对测试完成")
                    print(f"[{map_name}] 汇总数据已保存: {summary_csv_path}")

                    # 添加到场景结果
                    for stats_df in all_stats:
                        scene_results.append(stats_df)
                    success_count += 1

            except Exception as e:
                print(f"\n[{map_name}] ❌ 错误: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'='*70}")
        print(f"[{scene_name}] 场景完成: {success_count}/{len(map_files)} 个地图成功")
        print(f"{'='*70}")

        # 保存场景汇总
        if scene_results:
            combined_df = pd.concat(scene_results, ignore_index=True)

            summary_dir = Path(f"./results/{scene_name}/summary")
            summary_dir.mkdir(parents=True, exist_ok=True)

            summary_csv_path = summary_dir / "scene_summary.csv"
            combined_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
            print(f"[{scene_name}] 场景汇总已保存: {summary_csv_path}\n")

            all_scene_results[scene_name] = combined_df

    # 保存全局汇总
    if all_scene_results:
        print(f"\n{'#'*70}")
        print("所有场景汇总")
        print(f"{'#'*70}\n")

        global_summary_dir = Path("./results/global_summary")
        global_summary_dir.mkdir(parents=True, exist_ok=True)

        for scene_name, scene_df in all_scene_results.items():
            scene_df.insert(0, '场景', scene_name)

        global_df = pd.concat(all_scene_results.values(), ignore_index=True)
        global_csv_path = global_summary_dir / "all_scenes_summary.csv"
        global_df.to_csv(global_csv_path, index=False, encoding='utf-8')

        print(f"全局汇总已保存: {global_csv_path}")
        print("\n" + global_df.to_string(index=False))
        print(f"\n{'#'*70}")
        print(f"✅ 所有实验完成！共 {len(all_scene_results)} 个场景")
        print(f"{'#'*70}\n")
    else:
        print("\n❌ 所有场景测试均失败！")


def main():
    """主函数 - 支持多场景并行，自动检测map目录下的所有场景文件夹"""

    # 基础地图目录
    BASE_MAP_DIR = "/Users/yangzhaofeng/VsCodeProject/orchard_path_planning/experiment/map/"

    # 自动检测所有场景文件夹
    scene_configs = []

    if not os.path.exists(BASE_MAP_DIR):
        print(f"错误: 基础地图目录不存在: {BASE_MAP_DIR}")
        return

    # 扫描所有子文件夹
    for item in sorted(os.listdir(BASE_MAP_DIR)):
        item_path = os.path.join(BASE_MAP_DIR, item)
        # 只处理文件夹（排除文件）
        if os.path.isdir(item_path):
            scene_configs.append((item, item_path))

    if not scene_configs:
        print(f"错误: 在 {BASE_MAP_DIR} 中未找到任何场景文件夹")
        return

    print(f"\n{'='*70}")
    print(f"自动检测到 {len(scene_configs)} 个场景:")
    print(f"{'='*70}")
    for idx, (scene_name, scene_path) in enumerate(scene_configs, 1):
        print(f"  {idx}. {scene_name}")
        print(f"     路径: {scene_path}")
    print(f"{'='*70}\n")

    # 顺序运行实验（取消多进程）
    # use_random_poses=False: 使用地图文件中的起点终点（推荐，使用地图预生成的10对起点终点）
    # use_random_poses=True: 随机生成起点终点（忽略地图中保存的起点终点）
    run_sequential_experiments(
        scene_configs,
        use_random_poses=False,  # 设置为 False 使用地图中保存的多对起点终点
        random_seed=42
    )


def main_single_scene():
    """主函数 - 单场景模式（原始版本，支持随机起终点）"""
    import glob

    # 配置参数
    MAP_DIR = "/Users/yangzhaofeng/VsCodeProject/orchard_path_planning/experiment/map/generated_orchard_maps/scene_D_unstructured_corridor"
    USE_RANDOM_POSES = True  # 是否随机生成起点终点
    RANDOM_SEED = 42

    # 搜索所有.npz文件
    map_files = sorted(glob.glob(os.path.join(MAP_DIR, "*.npz")))

    if not map_files:
        print(f"错误: 在 {MAP_DIR} 中未找到.npz文件")
        return

    print(f"找到 {len(map_files)} 个地图文件:")
    for i, map_file in enumerate(map_files, 1):
        print(f"  {i}. {os.path.basename(map_file)}")
    print()

    # 车辆参数
    VEHICLE = VehicleGeometry(
        front_length=1.25,
        rear_length=0.35,
        width=0.8,
        safety_margin=0.18
    )

    # 对每个地图运行实验
    all_map_results = []

    for map_idx, map_path in enumerate(tqdm(map_files, desc="处理地图", ncols=100), 1):
        map_name = os.path.basename(map_path).replace('.npz', '')
        print(f"\n{'='*70}")
        print(f"[{map_idx}/{len(map_files)}] 测试地图: {map_name}")
        print(f"{'='*70}")

        # 为每个地图创建独立输出目录
        output_dir = f"./results/scene_D/{map_name}"

        try:
            # 加载地图并生成起点终点
            env, start_pose, goal_pose = load_or_generate_poses(
                map_path,
                VEHICLE,
                use_map_poses=not USE_RANDOM_POSES,
                seed=RANDOM_SEED + map_idx
            )

            print(f"起点: ({start_pose.x:.2f}, {start_pose.y:.2f}, {np.degrees(start_pose.yaw):.1f}°)")
            print(f"终点: ({goal_pose.x:.2f}, {goal_pose.y:.2f}, {np.degrees(goal_pose.yaw):.1f}°)")

            # 创建实验
            experiment = AblationExperiment(
                map_path=map_path,
                start_pose=start_pose,
                goal_pose=goal_pose,
                vehicle=VEHICLE,
                output_dir=output_dir,
                env=env
            )

            # 运行实验
            results_df = experiment.run_experiment(num_trials=50)

            # 保存原始数据
            results_csv_path = experiment.output_dir / "raw_results.csv"
            results_df.to_csv(results_csv_path, index=False, encoding='utf-8')
            print(f"\n原始数据已保存: {results_csv_path}")

            # 计算统计指标
            stats_df = experiment.compute_statistics(results_df)

            # 添加地图名称列
            stats_df.insert(0, '地图', map_name)

            # 保存起终点信息
            pose_info = pd.DataFrame([{
                '地图': map_name,
                '起点X': start_pose.x,
                '起点Y': start_pose.y,
                '起点Yaw': np.degrees(start_pose.yaw),
                '终点X': goal_pose.x,
                '终点Y': goal_pose.y,
                '终点Yaw': np.degrees(goal_pose.yaw),
                '直线距离': np.hypot(goal_pose.x - start_pose.x, goal_pose.y - start_pose.y)
            }])
            pose_csv_path = experiment.output_dir / "start_goal_poses.csv"
            pose_info.to_csv(pose_csv_path, index=False, encoding='utf-8')

            # 打印统计结果
            print("\n" + "="*70)
            print(f"地图 {map_name} 统计结果:")
            print("="*70)
            print(stats_df.to_string(index=False))
            print("="*70)

            # 保存统计数据
            stats_csv_path = experiment.output_dir / "statistics.csv"
            stats_df.to_csv(stats_csv_path, index=False, encoding='utf-8')
            print(f"\n统计数据已保存: {stats_csv_path}")

            # 收集结果
            all_map_results.append(stats_df)

        except Exception as e:
            print(f"\n错误: 地图 {map_name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 汇总所有地图的结果
    if all_map_results:
        print(f"\n{'='*70}")
        print("所有地图汇总结果")
        print(f"{'='*70}\n")

        combined_df = pd.concat(all_map_results, ignore_index=True)

        # 保存汇总结果
        summary_dir = Path("./results/summary")
        summary_dir.mkdir(parents=True, exist_ok=True)

        summary_csv_path = summary_dir / "all_maps_summary.csv"
        combined_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
        print(f"汇总结果已保存: {summary_csv_path}")

        # 打印汇总表格
        print("\n" + combined_df.to_string(index=False))
        print(f"\n{'='*70}")
        print(f"实验完成！共测试 {len(map_files)} 个地图")
        print(f"{'='*70}")
    else:
        print("\n所有地图测试均失败！")


if __name__ == "__main__":
    # 使用顺序执行模式（取消多进程）
    main()

    # 如果需要单场景模式，取消注释下面这行
    # main_single_scene()
