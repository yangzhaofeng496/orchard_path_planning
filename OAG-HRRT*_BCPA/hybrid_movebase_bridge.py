"""Complete bridge: map + tf + 2D Nav Goal → hybrid planning → move_base.

监听 /map, /tf, /move_base_simple/goal，使用 hybrid 算法规划后发布到 move_base。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
import uuid

import matplotlib.pyplot as plt
import numpy as np
import roslibpy
import yaml


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
INNOVATION_DIR = os.path.join(
    PROJECT_ROOT, "global_path_planning", "innovation_sample"
)
DEFAULT_MAP_NPZ = os.path.join(PROJECT_ROOT, "oag_hrrt_dwa", "orchard_simple.npz")

for import_dir in (INNOVATION_DIR, PROJECT_ROOT):
    if import_dir not in sys.path:
        sys.path.insert(0, import_dir)

from ackermann_rrt_star import AckermannRRTStar
from hybrid_sampler import SamplingCorridor
from vehicle.reeds_shepp_path import Pose
from vehicle.vehicle_collision import VehicleGeometry, CircleObstacle, check_pose_collision
# from RRT_start_optimize import (
from RRT_start_optimize_corridor import (
    DEFAULT_CONFIG_PATH,
    extract_corridor_info,
    finalize_reference_path,
    load_path_optimization_config,
    optimize_global_path,
)
from orchard_environment import load_environment

def yaw_from_quaternion(q):
    """从 ROS 四元数提取 yaw 角"""
    return math.atan2(
        2.0 * (q.get("w", 1.0) * q.get("z", 0.0) + q.get("x", 0.0) * q.get("y", 0.0)),
        1.0 - 2.0 * (q.get("y", 0.0) ** 2 + q.get("z", 0.0) ** 2),
    )


def quaternion_from_yaw(yaw):
    """从 yaw 角生成 ROS 四元数"""
    return {
        "x": 0.0,
        "y": 0.0,
        "z": math.sin(float(yaw) / 2.0),
        "w": math.cos(float(yaw) / 2.0),
    }


def ros_stamp():
    """生成 ROS 时间戳"""
    now = time.time()
    return {"secs": int(now), "nsecs": int((now - int(now)) * 1e9)}


class MapEnvironment:
    """从 ROS OccupancyGrid 转换的环境"""

    def __init__(self):
        self.obstacles = []
        self.bounds = (0, 100, 0, 100)
        self.resolution = 0.05
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.width = 0
        self.height = 0
        self.goal_pos = (0, 0)
        self.corridors = []
        self.goal_rectangle = None

    def update_from_map(self, map_msg):
        """从 OccupancyGrid 消息更新环境"""
        info = map_msg["info"]
        self.resolution = float(info["resolution"])
        self.width = int(info["width"])
        self.height = int(info["height"])

        origin = info["origin"]
        self.origin_x = float(origin["position"]["x"])
        self.origin_y = float(origin["position"]["y"])

        self.bounds = (
            self.origin_x,
            self.origin_x + self.width * self.resolution,
            self.origin_y,
            self.origin_y + self.height * self.resolution,
        )

        self.obstacles = self._extract_obstacles(map_msg["data"])
        print(f"[Map] 更新: {self.width}x{self.height}, "
              f"分辨率 {self.resolution}m, 障碍物 {len(self.obstacles)} 个")

    def update_from_npz(self, npz_path):
        """直接加载 NPZ 环境，避免将 OccupancyGrid 拆成大量障碍圆。"""
        environment = load_environment(npz_path)
        self.obstacles = list(environment.obstacles)
        self.bounds = tuple(environment.bounds)
        self.goal_pos = tuple(environment.goal_pos)
        self.corridors = list(environment.corridors)
        self.goal_rectangle = environment.goal_rectangle
        self.origin_x = float(self.bounds[0])
        self.origin_y = float(self.bounds[2])
        self.width = 0
        self.height = 0
        print(
            f"[Map] 已加载本地 NPZ: {npz_path} | "
            f"范围={self.bounds} | 障碍物={len(self.obstacles)} 个"
        )

    def _extract_obstacles(self, data):
        """从栅格地图提取障碍物"""
        obstacles = []
        threshold = 50
        # 增加采样步长，减少障碍物数量，加快碰撞检测
        sample_step = max(1, int(0.5 / self.resolution))  # 从0.2改为0.5
        # 使用采样步长的一半作为障碍物半径
        obstacle_radius = sample_step * self.resolution / 2.0

        for i in range(0, self.height, sample_step):
            for j in range(0, self.width, sample_step):
                idx = i * self.width + j
                if idx < len(data) and data[idx] > threshold:
                    world_x = self.origin_x + (j + 0.5) * self.resolution
                    world_y = self.origin_y + (i + 0.5) * self.resolution
                    obstacles.append(CircleObstacle(world_x, world_y, obstacle_radius))

        return obstacles


class GoalRectangle:
    """目标矩形区域"""

    def __init__(self):
        self.anchor_x = 0.0
        self.anchor_y = 0.0
        self.length = 30.0
        self.width = 22.0
        self.forward_offset = 0.0


class TFListener:
    """简化的 TF 监听器，支持多级 TF 树（如 map->odom->base_link）"""

    def __init__(self, ros, source_frame, target_frame):
        self.ros = ros
        self.source_frame = source_frame.lstrip('/')
        self.target_frame = target_frame.lstrip('/')
        self._lock = threading.Lock()
        self._transforms = {}  # 存储所有 TF: {(parent, child): transform}
        self._last_update = {}

        self.tf_sub = roslibpy.Topic(self.ros, "/tf", "tf2_msgs/TFMessage")
        self.tf_sub.subscribe(self._tf_callback)

    def _tf_callback(self, message):
        """TF 消息回调，存储所有变换"""
        for transform in message.get("transforms", []):
            header = transform.get("header", {})
            parent_frame = header.get("frame_id", "").lstrip('/')
            child_frame = transform.get("child_frame_id", "").lstrip('/')

            if parent_frame and child_frame:
                key = (parent_frame, child_frame)
                with self._lock:
                    self._transforms[key] = transform["transform"]
                    self._last_update[key] = time.time()

    def _multiply_transforms(self, t1, t2):
        """合成两个变换：t1 * t2"""
        # 提取平移
        x1, y1 = t1["translation"]["x"], t1["translation"]["y"]
        x2, y2 = t2["translation"]["x"], t2["translation"]["y"]

        # 提取旋转（四元数转欧拉角）
        yaw1 = yaw_from_quaternion(t1["rotation"])
        yaw2 = yaw_from_quaternion(t2["rotation"])

        # 合成变换
        cos_yaw1 = math.cos(yaw1)
        sin_yaw1 = math.sin(yaw1)

        x_new = x1 + cos_yaw1 * x2 - sin_yaw1 * y2
        y_new = y1 + sin_yaw1 * x2 + cos_yaw1 * y2
        yaw_new = yaw1 + yaw2

        return {
            "translation": {"x": x_new, "y": y_new, "z": 0.0},
            "rotation": quaternion_from_yaw(yaw_new)
        }

    def _find_transform_chain(self):
        """查找从 source_frame 到 target_frame 的变换链"""
        with self._lock:
            # 直接变换
            key_direct = (self.source_frame, self.target_frame)
            if key_direct in self._transforms:
                return [self._transforms[key_direct]]

            # 两级变换：source -> intermediate -> target
            # 尝试所有可能的中间帧
            for (parent1, child1), tf1 in self._transforms.items():
                if parent1 == self.source_frame:
                    # 找到 source -> intermediate
                    for (parent2, child2), tf2 in self._transforms.items():
                        if parent2 == child1 and child2 == self.target_frame:
                            # 找到 intermediate -> target
                            return [tf1, tf2]

            return None

    def lookup_transform(self, timeout=1.0):
        """查找变换，返回 (x, y, yaw) 或 None"""
        chain = self._find_transform_chain()
        if chain is None:
            return None

        # 检查超时
        with self._lock:
            current_time = time.time()
            for key, last_time in self._last_update.items():
                if current_time - last_time > timeout:
                    continue  # 允许部分超时

        # 合成变换链
        result_transform = chain[0]
        for tf in chain[1:]:
            result_transform = self._multiply_transforms(result_transform, tf)

        x = float(result_transform["translation"]["x"])
        y = float(result_transform["translation"]["y"])
        yaw = yaw_from_quaternion(result_transform["rotation"])

        return (x, y, yaw)

    def get_current_pose(self):
        """获取当前位姿作为 Pose 对象"""
        result = self.lookup_transform()
        if result is None:
            return None
        x, y, yaw = result
        return Pose(x, y, yaw)

    def close(self):
        """关闭订阅"""
        self.tf_sub.unsubscribe()


class HybridMapPlanner:
    """基于 map 的 hybrid 路径规划器"""

    def __init__(
        self,
        max_iterations=2500,
        seed=42,
        rectangle_length=30.0,
        rectangle_width=22.0,
        allow_reverse=False,
        use_goal_connector=True,
        enable_optimization=True,
        path_optimization_config=None,
        vehicle_geometry=None,
        wheel_base=2.5,
        max_steer_deg=30.0,
        use_ackermann_constraints=True,
        goal_probability=0.20,
        tangent_probability=0.10,
        adaptive_sampling_probabilities=False,
        smoothing_iterations=2,
    ):
        self.environment = MapEnvironment()
        self.max_iterations = int(max_iterations)
        self.seed = int(seed)
        self.rectangle_length = float(rectangle_length)
        self.rectangle_width = float(rectangle_width)
        self.allow_reverse = bool(allow_reverse)
        self.use_goal_connector = bool(use_goal_connector)
        self.enable_optimization = bool(enable_optimization)
        self.path_optimization_config = path_optimization_config or {}
        self.wheel_base = float(wheel_base)
        self.max_steer_deg = float(max_steer_deg)
        self.use_ackermann_constraints = bool(use_ackermann_constraints)
        self.goal_probability = float(goal_probability)
        self.tangent_probability = float(tangent_probability)
        self.adaptive_sampling_probabilities = bool(adaptive_sampling_probabilities)
        self.smoothing_iterations = int(smoothing_iterations)

        geometry = vehicle_geometry or {}
        self.vehicle = VehicleGeometry(
            front_length=float(geometry.get('front_length', 3.0)),
            rear_length=float(geometry.get('rear_length', 1.0)),
            width=float(geometry.get('width', 1.6)),
            safety_margin=float(geometry.get('safety_margin', 0.18)),
        )

        self._map_lock = threading.Lock()
        self._map_received = False

    def update_map(self, map_msg):
        """更新地图数据"""
        with self._map_lock:
            self.environment.update_from_map(map_msg)
            self._map_received = True

    def load_npz_map(self, npz_path):
        """加载本地 NPZ 地图并标记环境已就绪。"""
        with self._map_lock:
            self.environment.update_from_npz(npz_path)
            self._map_received = True

    def is_map_ready(self):
        """检查地图是否已接收"""
        with self._map_lock:
            return self._map_received

    def plan(self, start, goal, corridors=None):
        """执行 hybrid 路径规划"""
        with self._map_lock:
            if not self._map_received:
                raise ValueError("地图尚未接收")

            env = self.environment

            if check_pose_collision(start, self.vehicle, env.obstacles):
                raise ValueError("起始位姿与障碍物碰撞")
            if check_pose_collision(goal, self.vehicle, env.obstacles):
                raise ValueError("目标位姿与障碍物碰撞")

            env.goal_pos = (goal.x, goal.y)
            goal_rectangle = GoalRectangle()
            goal_rectangle.anchor_x = start.x
            goal_rectangle.anchor_y = start.y
            goal_rectangle.length = self.rectangle_length
            goal_rectangle.width = self.rectangle_width

            sampling_corridors = []
            corridor_data = corridors if corridors is not None else env.corridors
            if corridor_data:
                sampling_corridors = [
                    SamplingCorridor(c["x1"], c["y1"], c["x2"], c["y2"], c["width"])
                    for c in corridor_data
                ]

            # 车辆参数：与 oag_hrrt_dwa_demo.py 一致
            curvature = math.tan(math.radians(self.max_steer_deg)) / self.wheel_base

            planner = AckermannRRTStar(
                start=start,
                goal=goal,
                bounds=env.bounds,
                vehicle=self.vehicle,
                obstacles=env.obstacles,
                curvature=curvature,
                use_ackermann_constraints=self.use_ackermann_constraints,
                expand_length=3.0,
                step_size=0.08,
                max_iterations=self.max_iterations,
                near_radius=5.0,
                use_hybrid_sampling=True,
                corridors=sampling_corridors,
                goal_rectangle=goal_rectangle,
                rectangle_anchor_mode="closest_to_goal",
                goal_probability=self.goal_probability,
                tangent_probability=self.tangent_probability,
                adaptive_sampling_probabilities=self.adaptive_sampling_probabilities,
                corridor_probability=0.0,  # 与 oag_hrrt_dwa_demo.py 一致，不使用走廊
                rectangle_probability=0.45,
                allow_reverse=self.allow_reverse,
                use_tangent_guidance=True,
                shrink_probability=0.35,
                shrink_length_factor=0.70,
                shrink_width_factor=0.70,
                shrink_activation_distance=18.0,
                near_anchor_probability=0.55,
                near_anchor_length_ratio=0.40,
                cluster_shape="ellipse",
                use_goal_connector=self.use_goal_connector,
                relax_goal_yaw=False,
                random_seed=self.seed,
            )

        # 添加回调函数来打印规划进度
        def progress_callback(iteration):
            nodes_count = len(planner.nodes)
            best_cost = planner.best_cost if planner.best_cost < float('inf') else None
            goal_status = "✓ 找到" if planner.goal_index is not None else "搜索中"
            if best_cost is not None:
                print(f"[Plan] 迭代 {iteration}/{self.max_iterations} | 节点 {nodes_count} | 最佳代价 {best_cost:.2f} | 目标 {goal_status}")
            else:
                print(f"[Plan] 迭代 {iteration}/{self.max_iterations} | 节点 {nodes_count} | 目标 {goal_status}")

        print(f"[Plan] 开始规划，最大迭代次数 {self.max_iterations}，障碍物 {len(env.obstacles)} 个")
        result = planner.planning(callback=progress_callback, callback_interval=10)

        if result is None:
            print(f"[Plan] 规划失败，未找到路径")
            return []

        path_x, path_y, path_yaw = result[:3]
        print(f"[Plan] 规划成功！")
        print(f"[Plan] - 总迭代次数: {planner.first_solution_iteration if planner.first_solution_iteration else self.max_iterations}")
        print(f"[Plan] - 总节点数: {len(planner.nodes)}")
        print(f"[Plan] - 路径点数: {len(path_x)}")
        print(f"[Plan] - 最终代价: {planner.nodes[planner.goal_index].cost:.2f}")
        print(f"[Plan] - 规划耗时: {planner.planning_time:.2f}秒")

        # 路径优化（使用 RRT_start_optimize.py 中的策略）
        if self.enable_optimization:
            optimized_result = optimize_global_path(
                result=result,
                vehicle=self.vehicle,
                obstacles=env.obstacles,
                config=self.path_optimization_config,
                verbose=True,
            )
            if optimized_result is not None:
                result = optimized_result

        curvature_enabled = bool(
            self.path_optimization_config
            .get('curvature_smoothing', {})
            .get('enabled', True)
        )
        result = finalize_reference_path(
            result,
            goal_pose=goal,
            smoothing_iterations=self.smoothing_iterations,
            curvature_enabled=curvature_enabled,
        )
        path_x, path_y, path_yaw = result[:3]

        return [
            Pose(float(x), float(y), float(yaw))
            for x, y, yaw in zip(path_x, path_y, path_yaw)
        ]


class HybridMoveBaseBridge:
    """完整桥接: map + tf + 2D Nav Goal → hybrid planning → move_base"""

    def __init__(self, args):
        self.args = args
        self.ros = roslibpy.Ros(args.host, args.port)
        self.ros.run()
        if not self.ros.is_connected:
            raise RuntimeError(f"rosbridge 连接失败: {args.host}:{args.port}")

        with open(args.config, 'r', encoding='utf-8') as config_file:
            shared_config = yaml.safe_load(config_file) or {}
        planner_config = shared_config.get('planner', {})
        rectangle_config = planner_config.get('rectangle', {})
        vehicle_config = shared_config.get('vehicle', {})
        self._corridor_visual_cfg = (
            shared_config.get('corridor_alignment', {}).get('visualization', {})
        )
        geometry_config = vehicle_config.get('geometry', {})
        path_optimization_config = load_path_optimization_config(
            config_path=args.config,
            shortcut_iterations=args.shortcut_iterations,
            disable_curvature_smoothing=args.disable_curvature_smoothing,
        )

        max_iterations = (
            args.max_iterations if args.max_iterations is not None
            else planner_config.get('max_iterations', 2500)
        )
        seed = (
            args.seed if args.seed is not None
            else shared_config.get('random_seed', 42)
        )
        rectangle_length = (
            args.rectangle_length if args.rectangle_length is not None
            else rectangle_config.get('length', 30.0)
        )
        rectangle_width = (
            args.rectangle_width if args.rectangle_width is not None
            else rectangle_config.get('width', 22.0)
        )
        allow_reverse = (
            args.allow_reverse if args.allow_reverse is not None
            else planner_config.get('allow_reverse', False)
        )
        use_goal_connector = (
            not args.no_goal_connector if args.no_goal_connector is not None
            else planner_config.get('use_goal_connector', True)
        )
        enable_optimization = (
            args.enable_optimization if args.enable_optimization is not None
            else path_optimization_config.get('enabled', True)
        )

        # 初始化规划器
        self.planner = HybridMapPlanner(
            max_iterations=max_iterations,
            seed=seed,
            rectangle_length=rectangle_length,
            rectangle_width=rectangle_width,
            allow_reverse=allow_reverse,
            use_goal_connector=use_goal_connector,
            enable_optimization=enable_optimization,
            path_optimization_config=path_optimization_config,
            vehicle_geometry=geometry_config,
            wheel_base=vehicle_config.get('wheel_base', 2.5),
            max_steer_deg=vehicle_config.get('max_steer_deg', 30.0),
            use_ackermann_constraints=planner_config.get(
                'use_ackermann_constraints', True
            ),
            goal_probability=planner_config.get('goal_probability', 0.20),
            tangent_probability=planner_config.get('tangent_probability', 0.10),
            adaptive_sampling_probabilities=planner_config.get(
                'adaptive_sampling_probabilities', False
            ),
            smoothing_iterations=planner_config.get('smoothing_iterations', 2),
        )
        print(f"[Config] 已加载共享参数: {args.config}")

        # TF 监听器 (base_link 在 map 坐标系中的位置)
        self.tf_listener = TFListener(
            self.ros, args.map_frame, args.base_frame
        )

        # 优先使用本地 NPZ；未指定时才订阅 ROS OccupancyGrid。
        self.map_sub = None
        if args.map_npz:
            self.planner.load_npz_map(args.map_npz)
        else:
            self.map_sub = roslibpy.Topic(
                self.ros, args.map_topic, "nav_msgs/OccupancyGrid"
            )
            self.map_sub.subscribe(self._map_callback)

        # 订阅 2D Nav Goal
        self.goal_sub = roslibpy.Topic(
            self.ros, args.goal_topic, "geometry_msgs/PoseStamped"
        )
        self.goal_sub.subscribe(self._goal_callback)

        # 发布全局路径到 move_base
        self.path_pub = roslibpy.Topic(
            self.ros, args.path_topic, "nav_msgs/Path", latch=True
        )

        # 发布 move_base action goal
        self.action_goal_pub = roslibpy.Topic(
            self.ros, args.action_goal_topic, "move_base_msgs/MoveBaseActionGoal"
        )

        # 取消 action goal
        self.action_cancel_pub = roslibpy.Topic(
            self.ros, args.action_cancel_topic, "actionlib_msgs/GoalID"
        )

        # 暂停控制
        self._paused = False
        self._pause_lock = threading.Lock()
        self._current_goal_id = None
        self._request_number = 0

        if args.pause_topic:
            self.pause_sub = roslibpy.Topic(
                self.ros, args.pause_topic, "std_msgs/Bool"
            )
            self.pause_sub.subscribe(self._pause_callback)

        # 可视化设置
        self._enable_visualization = args.enable_visualization
        if self._enable_visualization:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互后端
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_title('Hybrid Navigation - Real-time Map & Path')
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self._map_image = None
            self._path_line = None
            self._robot_marker = None
            self._goal_marker = None
            self._view_set = False
            self._vis_output = args.vis_output
            self._static_patches = []
            print(f"  - 可视化: 已启用，保存到 {self._vis_output}")
            if args.map_npz:
                self._update_npz_visualization()

        print(f"[Bridge] Hybrid MoveBase 桥接已启动:")
        print(f"  - Map: {args.map_npz if args.map_npz else args.map_topic}")
        print(f"  - TF: {args.map_frame} -> {args.base_frame}")
        print(f"  - Goal: {args.goal_topic}")
        print(f"  - Path: {args.path_topic}")
        print(f"  - Action: {args.action_goal_topic}")
        if args.pause_topic:
            print(f"  - Pause: {args.pause_topic}")

    def _map_callback(self, message):
        """地图更新回调"""
        try:
            self.planner.update_map(message)
            if self._enable_visualization:
                self._update_map_visualization(message)
        except Exception as e:
            print(f"[Map] 更新失败: {e}")

    def _pause_callback(self, message):
        """暂停/恢复控制回调"""
        paused = message.get("data", False)
        with self._pause_lock:
            self._paused = paused

        if paused:
            print("[Bridge] 已暂停")
            if self._current_goal_id:
                self.action_cancel_pub.publish(roslibpy.Message({
                    "stamp": ros_stamp(),
                    "id": self._current_goal_id,
                }))
                print(f"[Bridge] 已取消目标: {self._current_goal_id}")
                self._current_goal_id = None
        else:
            print("[Bridge] 已恢复")

    def _goal_callback(self, message):
        """2D Nav Goal 回调"""
        with self._pause_lock:
            self._request_number += 1
            request_number = self._request_number
            paused = self._paused

        if paused:
            print("[Bridge] 暂停中，忽略 2D Nav Goal")
            return

        # 从 TF 获取当前位姿
        start = self.tf_listener.get_current_pose()
        if start is None:
            print(f"[Bridge] 无法获取 TF: {self.args.map_frame} -> {self.args.base_frame}")
            return

        # 解析目标位姿
        pose = message["pose"]
        position = pose["position"]
        goal = Pose(
            float(position["x"]),
            float(position["y"]),
            yaw_from_quaternion(pose["orientation"]),
        )

        frame_id = message.get("header", {}).get("frame_id") or self.args.map_frame

        # 异步规划
        threading.Thread(
            target=self._plan_and_publish,
            args=(request_number, start, goal, frame_id),
            daemon=True,
        ).start()

    def _plan_and_publish(self, request_number, start, goal, frame_id):
        """规划并发布路径"""
        print(f"[Bridge] 开始规划: ({start.x:.2f}, {start.y:.2f}, {math.degrees(start.yaw):.1f}°) -> "
              f"({goal.x:.2f}, {goal.y:.2f}, {math.degrees(goal.yaw):.1f}°)")

        try:
            path = self.planner.plan(start, goal)
        except Exception as e:
            print(f"[Bridge] 规划失败: {e}")
            return

        if not path:
            print("[Bridge] 规划失败，未找到路径")
            return

        print(f"[Bridge] 规划成功，共 {len(path)} 个路径点")

        # 记录请求的规划目标与最终路径终点之间的偏差。
        planned_endpoint = path[-1]
        delta_x = planned_endpoint.x - goal.x
        delta_y = planned_endpoint.y - goal.y
        position_error = math.hypot(delta_x, delta_y)
        yaw_error = math.atan2(
            math.sin(planned_endpoint.yaw - goal.yaw),
            math.cos(planned_endpoint.yaw - goal.yaw),
        )
        print(
            "[Plan] 目标点误差 | "
            f"规划目标=({goal.x:.3f}, {goal.y:.3f}, "
            f"{math.degrees(goal.yaw):.2f}°) | "
            f"路径终点=({planned_endpoint.x:.3f}, {planned_endpoint.y:.3f}, "
            f"{math.degrees(planned_endpoint.yaw):.2f}°) | "
            f"Δx={delta_x:.3f} m, Δy={delta_y:.3f} m, "
            f"位置误差={position_error:.3f} m, "
            f"航向误差={math.degrees(yaw_error):.2f}°"
        )

        # 可视化路径
        if self._enable_visualization:
            self._update_path_visualization(path, start, goal)

        # 检查是否有新目标
        with self._pause_lock:
            if request_number != self._request_number:
                print("[Bridge] 已收到新目标，丢弃旧规划")
                return
            if self._paused:
                print("[Bridge] 已暂停，不发布路径")
                return

        # 发布路径到 move_base
        stamp = ros_stamp()
        poses = []
        for pose in path:
            poses.append({
                "header": {"stamp": stamp, "frame_id": frame_id},
                "pose": {
                    "position": {"x": pose.x, "y": pose.y, "z": 0.0},
                    "orientation": quaternion_from_yaw(pose.yaw),
                },
            })

        self.path_pub.publish(roslibpy.Message({
            "header": {"stamp": stamp, "frame_id": frame_id},
            "poses": poses,
        }))
        print(f"[Bridge] 已发布全局路径到 {self.args.path_topic}")

        # 延迟激活 move_base
        if self.args.activation_delay > 0.0:
            time.sleep(self.args.activation_delay)

        with self._pause_lock:
            if self._paused:
                print("[Bridge] 已暂停，不激活 move_base")
                return

        # 发布 action goal
        goal_id = f"hybrid-{uuid.uuid4()}"
        target_pose = {
            "header": {"stamp": stamp, "frame_id": frame_id},
            "pose": {
                "position": {"x": goal.x, "y": goal.y, "z": 0.0},
                "orientation": quaternion_from_yaw(goal.yaw),
            },
        }

        self.action_goal_pub.publish(roslibpy.Message({
            "header": {"stamp": stamp, "frame_id": frame_id},
            "goal_id": {"stamp": stamp, "id": goal_id},
            "goal": {"target_pose": target_pose},
        }))

        with self._pause_lock:
            self._current_goal_id = goal_id

        print(f"[Bridge] 已激活 move_base: {goal_id}")

    def _update_map_visualization(self, map_msg):
        """更新地图可视化"""
        info = map_msg["info"]
        width = int(info["width"])
        height = int(info["height"])
        resolution = float(info["resolution"])
        origin_x = float(info["origin"]["position"]["x"])
        origin_y = float(info["origin"]["position"]["y"])

        # 转换为numpy数组
        data = np.array(map_msg["data"], dtype=np.int8).reshape(height, width)

        # 创建彩色地图
        map_rgb = np.zeros((height, width, 3))
        unknown = data == -1
        map_rgb[unknown] = [0.5, 0.5, 0.5]  # 灰色
        free = (data >= 0) & (data < 50)
        map_rgb[free] = 1.0  # 白色
        occupied = data >= 50
        intensity = 1.0 - (data[occupied] / 100.0)
        map_rgb[occupied] = np.stack([intensity, intensity, intensity], axis=-1)

        # 翻转Y轴
        map_rgb = np.flipud(map_rgb)

        # 地图范围
        width_m = width * resolution
        height_m = height * resolution
        extent = [origin_x, origin_x + width_m, origin_y, origin_y + height_m]

        if self._map_image is None:
            self._map_image = self.ax.imshow(
                map_rgb, extent=extent, origin='lower',
                interpolation='nearest', zorder=0
            )
        else:
            self._map_image.set_data(map_rgb)
            self._map_image.set_extent(extent)

        # 首次设置视图
        if not self._view_set:
            self.ax.set_xlim(origin_x, origin_x + width_m)
            self.ax.set_ylim(origin_y, origin_y + height_m)
            self._view_set = True

        # 保存图像
        self.fig.savefig(self._vis_output, dpi=100, bbox_inches='tight')
        print(f"[Vis] 地图已更新并保存到 {self._vis_output}")

    def _update_npz_visualization(self):
        """绘制本地 NPZ 中的圆形障碍物并保存图像。"""
        self._draw_npz_obstacles(None)

        env = self.planner.environment
        self.ax.set_xlim(env.bounds[0], env.bounds[1])
        self.ax.set_ylim(env.bounds[2], env.bounds[3])
        self._view_set = True
        if env.obstacles:
            self.ax.legend(loc='upper right')
        self.fig.savefig(self._vis_output, dpi=100, bbox_inches='tight')
        print(
            f"[Vis] NPZ地图已保存到 {self._vis_output} | "
            f"障碍物={len(env.obstacles)}"
        )

    def _draw_npz_obstacles(self, corridor_report):
        """按通道检测结果重绘NPZ圆形障碍物。"""
        for patch in self._static_patches:
            patch.remove()
        self._static_patches = []

        env = self.planner.environment
        all_corridor, optimized_corridor = extract_corridor_info(
            corridor_report, env.obstacles
        )
        failed_corridor = all_corridor - optimized_corridor
        for index, obstacle in enumerate(env.obstacles):
            if index in optimized_corridor:
                facecolor, edgecolor, linewidth = 'lightgreen', 'darkgreen', 2.5
                label = 'Optimized corridor' if index == min(optimized_corridor) else None
            elif index in failed_corridor:
                facecolor, edgecolor, linewidth = 'orange', 'darkorange', 2.5
                label = 'Detected corridor (not optimized)' if index == min(failed_corridor) else None
            else:
                facecolor, edgecolor, linewidth = 'lightcoral', 'red', 1.0
                label = 'Regular obstacle' if index == 0 else None
            patch = plt.Circle(
                (obstacle.x, obstacle.y),
                obstacle.radius,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=0.60,
                label=label,
                zorder=1,
            )
            self.ax.add_patch(patch)
            self._static_patches.append(patch)

    def _update_path_visualization(self, path, start, goal):
        """更新路径可视化"""
        # 清除旧的动态元素
        for artist in list(self.ax.lines):
            artist.remove()
        for artist in list(self.ax.patches):
            if artist not in self._static_patches:
                artist.remove()
        for artist in list(self.ax.texts):
            artist.remove()

        corridor_report = self.planner.path_optimization_config.get(
            '_corridor_report'
        )
        corridor_input = self.planner.path_optimization_config.get(
            '_corridor_input_path'
        )
        if self.args.map_npz:
            self._draw_npz_obstacles(corridor_report)

        # 绘制通道优化前后的路径
        if corridor_input is not None and len(corridor_input) >= 2:
            self.ax.plot(
                corridor_input[:, 0], corridor_input[:, 1],
                linestyle='--', color='royalblue',
                linewidth=float(self._corridor_visual_cfg.get(
                    'unoptimized_line_width', 1.8
                )), alpha=0.85,
                label='Before corridor optimization', zorder=2,
            )
        x_coords = [p.x for p in path]
        y_coords = [p.y for p in path]
        self.ax.plot(
            x_coords, y_coords, color='green', linestyle='-',
            linewidth=float(self._corridor_visual_cfg.get(
                'optimized_line_width', 2.5
            )), label='After corridor optimization', zorder=3,
        )

        # 绘制通道圆对连线、编号、净宽和成功状态。
        if corridor_report is not None:
            for number, pair_info in enumerate(corridor_report.get('pairs', []), 1):
                indices = pair_info.get('obstacle_indices', ())
                if len(indices) != 2:
                    continue
                first = self.planner.environment.obstacles[indices[0]]
                second = self.planner.environment.obstacles[indices[1]]
                valid = bool(pair_info.get('valid', False))
                color = 'darkgreen' if valid else 'darkorange'
                self.ax.plot(
                    [first.x, second.x], [first.y, second.y],
                    color=color, linestyle='-' if valid else '--',
                    linewidth=1.5, alpha=0.8 if valid else 0.6, zorder=2,
                )
                middle_x = 0.5 * (first.x + second.x)
                middle_y = 0.5 * (first.y + second.y)
                gap = math.hypot(second.x - first.x, second.y - first.y) \
                    - first.radius - second.radius
                self.ax.text(
                    middle_x, middle_y,
                    f"{'✓' if valid else '✗'} C{number}\n{gap:.1f}m",
                    color=color, fontsize=9, fontweight='bold',
                    ha='center', va='center', zorder=4,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor=color, alpha=0.85),
                )

        # 绘制起点
        self.ax.plot(start.x, start.y, 'bo', markersize=10, label='Start', zorder=3)

        # 绘制起点朝向箭头
        arrow_len = 1.5
        dx = arrow_len * np.cos(start.yaw)
        dy = arrow_len * np.sin(start.yaw)
        self.ax.arrow(
            start.x, start.y, dx, dy,
            head_width=0.5, head_length=0.5,
            fc='blue', ec='blue', zorder=3
        )

        # 绘制终点
        self.ax.plot(goal.x, goal.y, 'r*', markersize=15, label='Goal', zorder=3)

        # 绘制终点朝向箭头
        dx = arrow_len * np.cos(goal.yaw)
        dy = arrow_len * np.sin(goal.yaw)
        self.ax.arrow(
            goal.x, goal.y, dx, dy,
            head_width=0.5, head_length=0.5,
            fc='red', ec='red', zorder=3
        )

        self.ax.legend(loc='upper right')
        optimized_count = (
            corridor_report.get('optimized_pair_count', 0)
            if corridor_report is not None else 0
        )
        candidate_count = (
            corridor_report.get('candidate_pair_count', 0)
            if corridor_report is not None else 0
        )
        self.ax.set_title(
            f'Hybrid Navigation | Corridors: {optimized_count}/{candidate_count} optimized'
        )

        # 保存图像
        self.fig.savefig(self._vis_output, dpi=100, bbox_inches='tight')
        print(f"[Vis] 路径已更新并保存到 {self._vis_output}")

    def close(self):
        """关闭连接"""
        if self.map_sub is not None:
            self.map_sub.unsubscribe()
        self.goal_sub.unsubscribe()
        self.tf_listener.close()
        if hasattr(self, 'pause_sub'):
            self.pause_sub.unsubscribe()
        if self._enable_visualization:
            plt.close(self.fig)
        self.ros.terminate()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="RRT、车辆及路径优化的统一 YAML 配置",
    )

    # 连接参数
    parser.add_argument("--host", default="192.168.2.61", help="rosbridge 主机")
    parser.add_argument("--port", type=int, default=9090, help="rosbridge 端口")

    # 话题参数
    parser.add_argument(
        "--map-npz",
        default=DEFAULT_MAP_NPZ,
        help="本地 NPZ 地图；传空字符串时改为订阅 --map-topic",
    )
    parser.add_argument("--map-topic", default="/map", help="地图话题")
    parser.add_argument("--goal-topic", default="/move_base_simple/goal",
                        help="2D Nav Goal 话题")
    parser.add_argument("--path-topic", default="/external_global_plan",
                        help="全局路径发布话题")
    parser.add_argument("--action-goal-topic", default="/move_base/goal",
                        help="move_base action goal 话题")
    parser.add_argument("--action-cancel-topic", default="/move_base/cancel",
                        help="move_base cancel 话题")
    parser.add_argument("--pause-topic", default="",
                        help="暂停控制话题 (std_msgs/Bool)")

    # TF 参数
    parser.add_argument("--map-frame", default="map", help="地图坐标系")
    parser.add_argument("--base-frame", default="base_link", help="机器人基座坐标系")

    # 其他参数
    parser.add_argument("--activation-delay", type=float, default=0.2,
                        help="路径发布后延迟激活 move_base (秒)")
    parser.add_argument("--enable-visualization", action="store_true",
                        help="启用本地地图和路径可视化")
    parser.add_argument("--vis-output", default="hybrid_navigation.png",
                        help="可视化输出文件路径")

    # 规划器参数
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rectangle-length", type=float, default=None)
    parser.add_argument("--rectangle-width", type=float, default=None)
    parser.add_argument("--allow-reverse", action="store_true", default=None)
    parser.add_argument("--no-goal-connector", action="store_true", default=None)

    # 路径优化参数
    parser.add_argument(
        "--enable-optimization",
        action="store_true",
        default=None,
        help="启用路径优化 (Shortcut + 曲率平滑)"
    )
    parser.add_argument(
        "--no-optimization",
        dest="enable_optimization",
        action="store_false",
        default=None,
        help="禁用路径优化"
    )
    parser.add_argument("--shortcut-iterations", type=int, default=None,
                        help="Shortcut 优化迭代次数（默认使用 YAML）")
    parser.add_argument("--disable-curvature-smoothing", action="store_true",
                        help="禁用曲率平滑")

    return parser.parse_args()


def main():
    args = parse_args()
    bridge = HybridMoveBaseBridge(args)

    print("\n桥接已就绪，等待 2D Nav Goal...")
    if args.enable_visualization:
        print(f"可视化已启用，图像保存到: {args.vis_output}")
        print("每次规划后会自动更新图像")

    try:
        while bridge.ros.is_connected:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n正在退出...")
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
