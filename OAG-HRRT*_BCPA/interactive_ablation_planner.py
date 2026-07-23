#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
交互绘制起终点矩形，并进行 N×N RRT* 消融实验。

操作：
1. 左键拖拽绘制起点矩形；
2. 再次左键拖拽绘制终点矩形；
3. 按 Enter 生成安全采样点并开始实验；
4. 按 S 重新选择起点矩形；
5. 按 G 重新选择终点矩形；
6. 按 R 清空重置；
7. 按 Esc 退出。

消融配置：
1. Baseline：普通随机采样；
2. GoalBias：普通随机采样 + 目标偏置；
3. GoalBias+Tangent：普通随机采样 + 目标偏置 + 切向采样。

YAML示例：
planner:
  goal_probability: 0.20
  tangent_probability: 0.60
  adaptive_sampling_probabilities: false
"""

import argparse
import csv
import json
import math
import os
import secrets
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import RectangleSelector

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "STSong", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
INNOVATION_DIR = os.path.join(PROJECT_ROOT, "global_path_planning", "innovation_sample")
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "oag_hrrt_dwa", "config.yaml")

for path in (INNOVATION_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from global_path_planning.innovation_sample.ackermann_rrt_star import AckermannRRTStar
from global_path_planning.innovation_sample.hybrid_sampler import SamplingCorridor
from global_path_planning.innovation_sample.orchard_environment import load_environment, make_goal_rectangle
from vehicle.reeds_shepp_path import Pose
from vehicle.vehicle_collision import VehicleGeometry


@dataclass
class PlannerConfig:
    name: str
    color: str
    goal_probability: float
    tangent_probability: float
    use_tangent_guidance: bool
    adaptive_sampling_probabilities: bool
    description: str


@dataclass
class PlanningResult:
    pair_id: int
    start_index: int
    goal_index: int
    seed: int
    config_name: str
    success: bool
    start_x: float
    start_y: float
    start_yaw: float
    goal_x: float
    goal_y: float
    goal_yaw: float
    planning_time: float
    path_length: float
    num_nodes: int
    first_solution_iter: int
    num_path_points: int
    error: str = ""


class InteractiveRectangleExperiment:
    def __init__(
        self, npz_path, config_path, n=5, seed=None, obstacle_clearance=0.8,
        boundary_clearance=0.5, min_point_spacing=1.0, min_pair_distance=5.0,
        output_dir="rectangle_ablation_results", save_plots=True
    ):
        self.npz_path = str(npz_path)
        self.config_path = str(config_path)
        self.n = int(n)
        self.obstacle_clearance = float(obstacle_clearance)
        self.boundary_clearance = float(boundary_clearance)
        self.min_point_spacing = float(min_point_spacing)
        self.min_pair_distance = float(min_pair_distance)
        self.save_plots = bool(save_plots)

        with open(self.config_path, "r", encoding="utf-8") as file:
            self.yaml_config = yaml.safe_load(file) or {}

        print(f"[环境] 加载地图：{self.npz_path}")
        self.env = load_environment(self.npz_path)

        vehicle_cfg = self.yaml_config.get("vehicle", {})
        geometry_cfg = vehicle_cfg.get("geometry", {})
        planner_cfg = self.yaml_config.get("planner", {})
        rectangle_cfg = planner_cfg.get("rectangle", {})

        self.vehicle = VehicleGeometry(
            front_length=geometry_cfg.get("front_length", 3.0),
            rear_length=geometry_cfg.get("rear_length", 1.0),
            width=geometry_cfg.get("width", 1.6),
            safety_margin=geometry_cfg.get("safety_margin", 0.18),
        )

        self.min_turning_radius = float(vehicle_cfg.get("min_turning_radius", 3.0))
        self.max_iterations = int(planner_cfg.get("max_iterations", 2500))
        self.rectangle_length = float(rectangle_cfg.get("length", 14.0))
        self.rectangle_width = float(rectangle_cfg.get("width", 9.0))

        self.goal_probability = float(planner_cfg.get("goal_probability", 0.20))
        self.tangent_probability = float(planner_cfg.get("tangent_probability", 0.60))
        self.adaptive_sampling_probabilities = bool(
            planner_cfg.get("adaptive_sampling_probabilities", False)
        )

        if not 0.0 <= self.goal_probability <= 1.0:
            raise ValueError("planner.goal_probability必须在0～1之间")
        if not 0.0 <= self.tangent_probability <= 1.0:
            raise ValueError("planner.tangent_probability必须在0～1之间")
        if not self.adaptive_sampling_probabilities:
            total_probability = self.goal_probability + self.tangent_probability
            if total_probability > 1.0 + 1e-9:
                raise ValueError(
                    "固定概率模式下，goal_probability + tangent_probability不能超过1.0，"
                    f"当前为{total_probability:.2f}"
                )

        self.ablation_configs = [
            PlannerConfig(
                name="Baseline",
                color="royalblue",
                goal_probability=0.0,
                tangent_probability=0.0,
                use_tangent_guidance=False,
                adaptive_sampling_probabilities=False,
                description="Baseline RRT*",
            ),
            PlannerConfig(
                name="GoalBias",
                color="darkorange",
                goal_probability=self.goal_probability,
                tangent_probability=0.0,
                use_tangent_guidance=False,
                adaptive_sampling_probabilities=False,
                description=f"RRT* + Goal Bias ({self.goal_probability:.0%})",
            ),
            PlannerConfig(
                name="GoalBias+Tangent",
                color="darkred",
                goal_probability=self.goal_probability,
                tangent_probability=self.tangent_probability,
                use_tangent_guidance=True,
                adaptive_sampling_probabilities=self.adaptive_sampling_probabilities,
                description=(
                    f"RRT* + Goal Bias ({self.goal_probability:.0%}) "
                    f"+ Tangent ({self.tangent_probability:.0%})"
                ),
            ),
        ]

        self.master_seed = int(seed) if seed is not None else secrets.randbits(63)
        self.rng = np.random.default_rng(self.master_seed)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_name = Path(self.npz_path).stem
        output_root = Path(output_dir)
        if not output_root.is_absolute():
            output_root = Path(CURRENT_DIR) / output_root

        self.result_dir = output_root / f"{map_name}_{timestamp}"
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.start_rect = None
        self.goal_rect = None
        self.start_points = []
        self.goal_points = []
        self.results = []
        self.paths = {config.name: [] for config in self.ablation_configs}

        self.active_region = "start"
        self.selection_confirmed = False
        self.fig = None
        self.ax = None
        self.selector = None
        self.start_patch = None
        self.goal_patch = None
        self.start_artists = []
        self.goal_artists = []

        print(f"[环境] 障碍物数量：{len(self.env.obstacles)}")
        print(f"[实验] N={self.n}，点对数量={self.n * self.n}，总规划次数={len(self.ablation_configs) * self.n * self.n}")
        print(f"[安全] 采样点到障碍物边缘的最小距离={self.obstacle_clearance:.2f}m")
        print(f"[安全] 障碍物中心最小距离=障碍半径+{self.obstacle_clearance:.2f}m")
        print(f"[随机] 主随机种子={self.master_seed}")
        print(
            f"[采样] goal={self.goal_probability:.2f}，"
            f"tangent={self.tangent_probability:.2f}，"
            f"adaptive={self.adaptive_sampling_probabilities}"
        )

    def draw_obstacles(self, ax, show_clearance=False):
        for obstacle in self.env.obstacles:
            ax.add_patch(Circle(
                (obstacle.x, obstacle.y), obstacle.radius,
                facecolor="lightcoral", edgecolor="red", alpha=0.60
            ))

            if show_clearance:
                ax.add_patch(Circle(
                    (obstacle.x, obstacle.y),
                    obstacle.radius + self.obstacle_clearance,
                    fill=False, edgecolor="orange", linestyle="--",
                    linewidth=0.9, alpha=0.45
                ))

    def setup_selection_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(13, 9))
        self.draw_obstacles(self.ax, show_clearance=True)

        self.ax.set_xlim(self.env.bounds[0], self.env.bounds[1])
        self.ax.set_ylim(self.env.bounds[2], self.env.bounds[3])
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X（米）")
        self.ax.set_ylabel("Y（米）")

        self.selector = RectangleSelector(
            self.ax, self.on_rectangle_selected, useblit=True,
            button=[1], minspanx=0.5, minspany=0.5,
            spancoords="data", interactive=False,
            props=dict(facecolor="gray", edgecolor="black", alpha=0.25)
        )

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.update_title()

    def update_title(self, message=None):
        if message is None:
            active = "起点矩形" if self.active_region == "start" else "终点矩形"
            message = (
                f"当前绘制：{active} | 左键拖拽绘制 | "
                "S：起点矩形 | G：终点矩形 | Enter：开始 | R：重置 | Esc：退出"
            )

        self.ax.set_title(message, fontsize=11)
        self.fig.canvas.draw_idle()

    def clip_rectangle(self, rectangle):
        x1, x2, y1, y2 = rectangle
        x_min, x_max, y_min, y_max = self.env.bounds
        x1 = max(x_min, min(x_max, x1))
        x2 = max(x_min, min(x_max, x2))
        y1 = max(y_min, min(y_max, y1))
        y2 = max(y_min, min(y_max, y2))
        return min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)

    def on_rectangle_selected(self, press_event, release_event):
        if press_event.xdata is None or release_event.xdata is None:
            return
        if press_event.ydata is None or release_event.ydata is None:
            return

        rectangle = self.clip_rectangle((
            press_event.xdata, release_event.xdata,
            press_event.ydata, release_event.ydata
        ))

        if rectangle[1] - rectangle[0] < 0.5 or rectangle[3] - rectangle[2] < 0.5:
            self.update_title("矩形太小，请重新绘制")
            return

        if self.active_region == "start":
            self.start_rect = rectangle
            self.active_region = "goal"
            print(f"[矩形] 起点范围：{rectangle}")
        else:
            self.goal_rect = rectangle
            print(f"[矩形] 终点范围：{rectangle}")

        self.redraw_rectangles()
        self.update_title()

    def redraw_rectangles(self):
        if self.start_patch is not None:
            self.start_patch.remove()
            self.start_patch = None
        if self.goal_patch is not None:
            self.goal_patch.remove()
            self.goal_patch = None

        if self.start_rect is not None:
            x1, x2, y1, y2 = self.start_rect
            self.start_patch = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor="green", linewidth=2.5, linestyle="--"
            )
            self.ax.add_patch(self.start_patch)

        if self.goal_rect is not None:
            x1, x2, y1, y2 = self.goal_rect
            self.goal_patch = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor="red", linewidth=2.5, linestyle="--"
            )
            self.ax.add_patch(self.goal_patch)

        self.fig.canvas.draw_idle()

    def clear_point_artists(self):
        for artist in self.start_artists + self.goal_artists:
            try:
                artist.remove()
            except ValueError:
                pass

        self.start_artists.clear()
        self.goal_artists.clear()

    def on_key_press(self, event):
        if event.key in ("s", "S"):
            self.active_region = "start"
            self.update_title()

        elif event.key in ("g", "G"):
            self.active_region = "goal"
            self.update_title()

        elif event.key in ("r", "R"):
            self.start_rect = None
            self.goal_rect = None
            self.start_points = []
            self.goal_points = []
            self.active_region = "start"
            self.clear_point_artists()
            self.redraw_rectangles()
            self.update_title("已重置，请绘制起点矩形")

        elif event.key in ("enter", "return"):
            if self.start_rect is None or self.goal_rect is None:
                self.update_title("请先绘制起点矩形和终点矩形")
                return

            try:
                self.generate_points()
                self.draw_generated_points()
                self.selection_confirmed = True
                self.update_title("采样点生成完成，即将开始N×N规划")
                self.fig.canvas.draw_idle()
                plt.pause(1.0)
                self.fig.savefig(
                    self.result_dir / "selected_rectangles_and_points.png",
                    dpi=160, bbox_inches="tight"
                )
                plt.close(self.fig)

            except Exception as error:
                print(f"[采样失败] {error}")
                self.update_title(f"采样失败：{error}")

        elif event.key == "escape":
            plt.close(self.fig)

    def point_is_safe(self, x, y):
        x_min, x_max, y_min, y_max = self.env.bounds

        if x < x_min + self.boundary_clearance or x > x_max - self.boundary_clearance:
            return False
        if y < y_min + self.boundary_clearance or y > y_max - self.boundary_clearance:
            return False

        for obstacle in self.env.obstacles:
            distance = math.hypot(x - obstacle.x, y - obstacle.y)
            if distance <= obstacle.radius + self.obstacle_clearance:
                return False

        return True

    def sample_points(self, rectangle, count, existing=None, opposite=None):
        x_min, x_max, y_min, y_max = rectangle
        existing = [] if existing is None else list(existing)
        opposite = [] if opposite is None else list(opposite)
        points = []
        max_attempts = max(20000, count * 10000)

        for _ in range(max_attempts):
            if len(points) >= count:
                break

            x = float(self.rng.uniform(x_min, x_max))
            y = float(self.rng.uniform(y_min, y_max))

            if not self.point_is_safe(x, y):
                continue

            if any(
                math.hypot(x - px, y - py) < self.min_point_spacing
                for px, py in existing + points
            ):
                continue

            if opposite and any(
                math.hypot(x - px, y - py) < self.min_pair_distance
                for px, py in opposite
            ):
                continue

            points.append((x, y))

        if len(points) != count:
            raise RuntimeError(
                f"矩形内只能生成{len(points)}/{count}个安全点。"
                "请扩大矩形、减小N或减小--obstacle-clearance。"
            )

        return points

    def generate_points(self):
        self.start_points = self.sample_points(self.start_rect, self.n)
        self.goal_points = self.sample_points(
            self.goal_rect, self.n, opposite=self.start_points
        )

        print("\n[起点采样]")
        for index, point in enumerate(self.start_points):
            print(f"  S{index}: ({point[0]:.2f}, {point[1]:.2f})")

        print("[终点采样]")
        for index, point in enumerate(self.goal_points):
            print(f"  G{index}: ({point[0]:.2f}, {point[1]:.2f})")

    def draw_generated_points(self):
        self.clear_point_artists()

        for index, (x, y) in enumerate(self.start_points):
            point_artist, = self.ax.plot(x, y, "go", markersize=8, zorder=10)
            text_artist = self.ax.text(x, y, f"S{index}", color="green", fontsize=9, zorder=10)
            self.start_artists.extend([point_artist, text_artist])

        for index, (x, y) in enumerate(self.goal_points):
            point_artist, = self.ax.plot(x, y, "r*", markersize=12, zorder=10)
            text_artist = self.ax.text(x, y, f"G{index}", color="red", fontsize=9, zorder=10)
            self.goal_artists.extend([point_artist, text_artist])

        self.fig.canvas.draw_idle()

    def select_rectangles(self):
        self.setup_selection_plot()
        plt.show()

        if not self.selection_confirmed:
            raise RuntimeError("未完成起终点矩形选择")

    @staticmethod
    def normalize_angle(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def build_pair_poses(self, start_point, goal_point):
        sx, sy = start_point
        gx, gy = goal_point
        yaw = math.atan2(gy - sy, gx - sx)

        return (
            Pose(sx, sy, self.normalize_angle(yaw)),
            Pose(gx, gy, self.normalize_angle(yaw))
        )

    def make_corridors(self):
        corridors = []

        for corridor in self.env.corridors:
            if isinstance(corridor, dict):
                corridors.append(SamplingCorridor(
                    corridor["x1"], corridor["y1"],
                    corridor["x2"], corridor["y2"],
                    corridor["width"]
                ))
            else:
                corridors.append(SamplingCorridor(
                    corridor.x1, corridor.y1,
                    corridor.x2, corridor.y2,
                    corridor.width
                ))

        return corridors

    @staticmethod
    def get_node_count(planner):
        if planner is None:
            return 0

        nodes = getattr(planner, "nodes", None)
        if nodes is None:
            nodes = getattr(planner, "node_list", [])

        return len(nodes)

    def plan_single(self, pair_id, start_index, goal_index, seed, start_pose, goal_pose, config):
        planner = None
        start_time = time.perf_counter()

        try:
            goal_rectangle = make_goal_rectangle(
                (start_pose.x, start_pose.y),
                (goal_pose.x, goal_pose.y),
                self.rectangle_length,
                self.rectangle_width,
                forward_offset=0.0
            )

            use_hybrid_sampling = (
                config.goal_probability > 0.0
                or config.tangent_probability > 0.0
            )
            enable_tangent_connectors = config.use_tangent_guidance

            planner = AckermannRRTStar(
                start=start_pose,
                goal=goal_pose,
                bounds=self.env.bounds,
                vehicle=self.vehicle,
                obstacles=self.env.obstacles,
                curvature=1.0 / self.min_turning_radius,
                # 所有消融组统一使用经过碰撞检测的直线段连接。
                use_ackermann_constraints=False,
                expand_length=3.0,
                step_size=0.08,
                max_iterations=self.max_iterations,
                near_radius=5.0,
                use_hybrid_sampling=use_hybrid_sampling,
                corridors=self.make_corridors(),
                goal_rectangle=goal_rectangle,
                rectangle_anchor_mode="closest_to_goal",
                goal_probability=config.goal_probability,
                tangent_probability=config.tangent_probability,
                adaptive_sampling_probabilities=config.adaptive_sampling_probabilities,
                corridor_probability=0.0,
                rectangle_probability=0.45,
                allow_reverse=enable_tangent_connectors,
                use_tangent_guidance=config.use_tangent_guidance,
                shrink_probability=0.35,
                shrink_length_factor=0.70,
                shrink_width_factor=0.70,
                shrink_activation_distance=18.0,
                near_anchor_probability=0.55,
                near_anchor_length_ratio=0.40,
                cluster_shape="ellipse",
                use_goal_connector=enable_tangent_connectors,
                relax_goal_yaw=False,
                random_seed=seed
            )

            raw_result = planner.planning()
            planning_time = time.perf_counter() - start_time
            num_nodes = self.get_node_count(planner)
            first_iteration = int(
                getattr(planner, "first_solution_iteration", 0) or 0
            )

            if raw_result is None:
                return PlanningResult(
                    pair_id, start_index, goal_index, seed, config.name, False,
                    start_pose.x, start_pose.y, start_pose.yaw,
                    goal_pose.x, goal_pose.y, goal_pose.yaw,
                    planning_time, 0.0, num_nodes, 0, 0
                ), None

            path_x, path_y, path_yaw = map(list, raw_result[:3])
            path_length = sum(
                math.hypot(
                    path_x[index + 1] - path_x[index],
                    path_y[index + 1] - path_y[index]
                )
                for index in range(len(path_x) - 1)
            )

            record = PlanningResult(
                pair_id, start_index, goal_index, seed, config.name, True,
                start_pose.x, start_pose.y, start_pose.yaw,
                goal_pose.x, goal_pose.y, goal_pose.yaw,
                planning_time, path_length, num_nodes,
                first_iteration, len(path_x)
            )

            path = {
                "pair_id": pair_id,
                "start_index": start_index,
                "goal_index": goal_index,
                "x": path_x,
                "y": path_y,
                "yaw": path_yaw
            }

            return record, path

        except Exception as error:
            planning_time = time.perf_counter() - start_time
            num_nodes = self.get_node_count(planner)

            return PlanningResult(
                pair_id, start_index, goal_index, seed, config.name, False,
                start_pose.x, start_pose.y, start_pose.yaw,
                goal_pose.x, goal_pose.y, goal_pose.yaw,
                planning_time, 0.0, num_nodes, 0, 0, str(error)
            ), None

    def run_all_pairs(self):
        total_pairs = self.n * self.n
        pair_id = 0

        for start_index, start_point in enumerate(self.start_points):
            for goal_index, goal_point in enumerate(self.goal_points):
                pair_seed = int(self.rng.integers(0, 2**31 - 1))
                start_pose, goal_pose = self.build_pair_poses(start_point, goal_point)

                print("\n" + "=" * 80)
                print(
                    f"[点对 {pair_id + 1}/{total_pairs}] "
                    f"S{start_index} → G{goal_index}，seed={pair_seed}"
                )

                for config in self.ablation_configs:
                    result, path = self.plan_single(
                        pair_id, start_index, goal_index, pair_seed,
                        start_pose, goal_pose, config
                    )

                    self.results.append(result)

                    if path is not None:
                        self.paths[config.name].append(path)

                    if result.success:
                        print(
                            f"  {config.name:22s} 成功 | "
                            f"时间={result.planning_time:.3f}s | "
                            f"长度={result.path_length:.2f}m | "
                            f"节点={result.num_nodes} | "
                            f"首次解={result.first_solution_iter}"
                        )
                    else:
                        error_text = f" | {result.error}" if result.error else ""
                        print(
                            f"  {config.name:22s} 失败 | "
                            f"时间={result.planning_time:.3f}s | "
                            f"节点={result.num_nodes}{error_text}"
                        )

                pair_id += 1

    @staticmethod
    def statistics(values):
        values = np.asarray(values, dtype=float)

        if len(values) == 0:
            return {
                "mean": None,
                "std": None,
                "median": None,
                "min": None,
                "max": None
            }

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }

    def build_summary(self):
        summaries = []
        total_pairs = self.n * self.n

        for config in self.ablation_configs:
            records = [
                result for result in self.results
                if result.config_name == config.name
            ]
            successes = [result for result in records if result.success]

            summaries.append({
                "config_name": config.name,
                "description": config.description,
                "goal_probability": config.goal_probability,
                "tangent_probability": config.tangent_probability,
                "adaptive_sampling_probabilities": config.adaptive_sampling_probabilities,
                "total_pairs": total_pairs,
                "success_count": len(successes),
                "failure_count": total_pairs - len(successes),
                "success_rate": len(successes) / total_pairs,
                "planning_time_all": self.statistics([
                    result.planning_time for result in records
                ]),
                "planning_time_success": self.statistics([
                    result.planning_time for result in successes
                ]),
                "path_length": self.statistics([
                    result.path_length for result in successes
                ]),
                "num_nodes_all": self.statistics([
                    result.num_nodes for result in records
                ]),
                "num_nodes_success": self.statistics([
                    result.num_nodes for result in successes
                ]),
                "first_solution_iter": self.statistics([
                    result.first_solution_iter
                    for result in successes
                    if result.first_solution_iter > 0
                ])
            })

        return summaries

    def save_results(self, summaries):
        if not self.results:
            raise RuntimeError("没有可保存的实验结果")

        detail_path = self.result_dir / "pair_results.csv"

        with detail_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(asdict(self.results[0]).keys())
            )
            writer.writeheader()
            writer.writerows(asdict(result) for result in self.results)

        summary_path = self.result_dir / "summary.csv"
        summary_fields = [
            "config_name",
            "goal_probability",
            "tangent_probability",
            "adaptive_sampling_probabilities",
            "total_pairs",
            "success_count",
            "failure_count",
            "success_rate",
            "mean_time_all",
            "std_time_all",
            "mean_time_success",
            "std_time_success",
            "mean_path_length",
            "std_path_length",
            "mean_nodes_all",
            "mean_nodes_success",
            "mean_first_solution_iter",
            "std_first_solution_iter"
        ]

        with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=summary_fields)
            writer.writeheader()

            for item in summaries:
                writer.writerow({
                    "config_name": item["config_name"],
                    "goal_probability": item["goal_probability"],
                    "tangent_probability": item["tangent_probability"],
                    "adaptive_sampling_probabilities": item["adaptive_sampling_probabilities"],
                    "total_pairs": item["total_pairs"],
                    "success_count": item["success_count"],
                    "failure_count": item["failure_count"],
                    "success_rate": item["success_rate"],
                    "mean_time_all": item["planning_time_all"]["mean"],
                    "std_time_all": item["planning_time_all"]["std"],
                    "mean_time_success": item["planning_time_success"]["mean"],
                    "std_time_success": item["planning_time_success"]["std"],
                    "mean_path_length": item["path_length"]["mean"],
                    "std_path_length": item["path_length"]["std"],
                    "mean_nodes_all": item["num_nodes_all"]["mean"],
                    "mean_nodes_success": item["num_nodes_success"]["mean"],
                    "mean_first_solution_iter": item["first_solution_iter"]["mean"],
                    "std_first_solution_iter": item["first_solution_iter"]["std"]
                })

        experiment = {
            "map": self.npz_path,
            "config": self.config_path,
            "timestamp": datetime.now().isoformat(),
            "master_seed": self.master_seed,
            "n": self.n,
            "total_pairs": self.n * self.n,
            "total_planner_runs": len(self.ablation_configs) * self.n * self.n,
            "start_rectangle": self.start_rect,
            "goal_rectangle": self.goal_rect,
            "start_points": self.start_points,
            "goal_points": self.goal_points,
            "obstacle_clearance": self.obstacle_clearance,
            "boundary_clearance": self.boundary_clearance,
            "min_point_spacing": self.min_point_spacing,
            "min_pair_distance": self.min_pair_distance,
            "planner_configs": [asdict(config) for config in self.ablation_configs],
            "summaries": summaries
        }

        json_path = self.result_dir / "experiment.json"

        with json_path.open("w", encoding="utf-8") as file:
            json.dump(experiment, file, ensure_ascii=False, indent=2)

        print(f"[保存] 详细结果：{detail_path}")
        print(f"[保存] 汇总结果：{summary_path}")
        print(f"[保存] 实验配置：{json_path}")

    def draw_base(self, ax):
        self.draw_obstacles(ax, show_clearance=False)

        if self.start_rect is not None:
            x1, x2, y1, y2 = self.start_rect
            ax.add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor="green", linewidth=2.0,
                linestyle="--", label="起点区域"
            ))

        if self.goal_rect is not None:
            x1, x2, y1, y2 = self.goal_rect
            ax.add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor="red", linewidth=2.0,
                linestyle="--", label="终点区域"
            ))

        for index, (x, y) in enumerate(self.start_points):
            ax.plot(x, y, "go", markersize=7)
            ax.text(x, y, f"S{index}", color="green", fontsize=8)

        for index, (x, y) in enumerate(self.goal_points):
            ax.plot(x, y, "r*", markersize=10)
            ax.text(x, y, f"G{index}", color="red", fontsize=8)

        ax.set_xlim(self.env.bounds[0], self.env.bounds[1])
        ax.set_ylim(self.env.bounds[2], self.env.bounds[3])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X（米）")
        ax.set_ylabel("Y（米）")

    def save_path_plots(self, summaries):
        if not self.save_plots:
            return

        for config in self.ablation_configs:
            figure, ax = plt.subplots(figsize=(12, 8))
            self.draw_base(ax)

            for path in self.paths[config.name]:
                ax.plot(
                    path["x"], path["y"],
                    color=config.color,
                    linewidth=1.2,
                    alpha=0.30
                )

            summary = next(
                item for item in summaries
                if item["config_name"] == config.name
            )

            ax.set_title(
                f"{config.description}\n"
                f"成功率={summary['success_rate'] * 100:.1f}% "
                f"({summary['success_count']}/{summary['total_pairs']})"
            )

            ax.legend(loc="upper right")
            figure.tight_layout()

            safe_name = config.name.replace("+", "_").replace(" ", "_")
            figure.savefig(
                self.result_dir / f"paths_{safe_name}.png",
                dpi=160,
                bbox_inches="tight"
            )
            plt.close(figure)

        names = [item["config_name"] for item in summaries]
        success_rates = [item["success_rate"] * 100.0 for item in summaries]
        mean_times = [
            item["planning_time_success"]["mean"] or 0.0
            for item in summaries
        ]
        mean_iterations = [
            item["first_solution_iter"]["mean"] or 0.0
            for item in summaries
        ]

        figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))

        axes[0].bar(names, success_rates)
        axes[0].set_title("规划成功率")
        axes[0].set_ylabel("成功率（%）")
        axes[0].set_ylim(0, 105)

        axes[1].bar(names, mean_times)
        axes[1].set_title("成功案例平均规划时间")
        axes[1].set_ylabel("时间（秒）")

        axes[2].bar(names, mean_iterations)
        axes[2].set_title("平均首次解迭代")
        axes[2].set_ylabel("迭代次数")

        for ax in axes:
            ax.tick_params(axis="x", rotation=15)
            ax.grid(True, axis="y", alpha=0.3)

        figure.tight_layout()
        figure.savefig(
            self.result_dir / "summary_comparison.png",
            dpi=160,
            bbox_inches="tight"
        )
        plt.close(figure)

    def print_summary(self, summaries):
        print("\n" + "=" * 115)
        print("N×N消融实验汇总")
        print("=" * 115)

        for item in summaries:
            mean_time = item["planning_time_success"]["mean"]
            mean_length = item["path_length"]["mean"]
            mean_nodes = item["num_nodes_success"]["mean"]
            mean_iteration = item["first_solution_iter"]["mean"]

            print(
                f"{item['config_name']:22s} | "
                f"成功率={item['success_rate'] * 100:6.2f}% | "
                f"成功={item['success_count']:3d}/{item['total_pairs']:3d} | "
                f"时间={mean_time if mean_time is not None else float('nan'):7.3f}s | "
                f"长度={mean_length if mean_length is not None else float('nan'):7.2f}m | "
                f"节点={mean_nodes if mean_nodes is not None else float('nan'):8.1f} | "
                f"首次解={mean_iteration if mean_iteration is not None else float('nan'):8.1f}"
            )

        print("=" * 115)

    def run(self):
        self.select_rectangles()
        self.run_all_pairs()
        summaries = self.build_summary()
        self.print_summary(summaries)
        self.save_results(summaries)
        self.save_path_plots(summaries)
        print(f"\n实验完成，结果目录：{self.result_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="交互绘制起终点矩形的N×N RRT*消融实验"
    )

    parser.add_argument("map", help="NPZ地图文件路径")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="YAML配置文件路径"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="起点矩形和终点矩形内分别生成的点数"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="不设置时每次运行自动生成新随机种子"
    )
    parser.add_argument(
        "--obstacle-clearance",
        type=float,
        default=0.8,
        help="采样点到障碍物边缘的最小距离，单位m"
    )
    parser.add_argument(
        "--boundary-clearance",
        type=float,
        default=0.5,
        help="采样点到地图边界的最小距离，单位m"
    )
    parser.add_argument(
        "--min-point-spacing",
        type=float,
        default=1.0,
        help="同一矩形内采样点之间的最小距离"
    )
    parser.add_argument(
        "--min-pair-distance",
        type=float,
        default=5.0,
        help="任一起点与任一终点之间的最小距离"
    )
    parser.add_argument(
        "--output-dir",
        default="rectangle_ablation_results",
        help="结果输出根目录"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="不保存路径可视化图片"
    )

    args = parser.parse_args()

    if not os.path.exists(args.map):
        raise FileNotFoundError(f"地图不存在：{args.map}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在：{args.config}")
    if args.n <= 0:
        raise ValueError("--n必须大于0")
    if args.obstacle_clearance < 0:
        raise ValueError("--obstacle-clearance不能小于0")
    if args.boundary_clearance < 0:
        raise ValueError("--boundary-clearance不能小于0")
    if args.min_point_spacing < 0:
        raise ValueError("--min-point-spacing不能小于0")
    if args.min_pair_distance < 0:
        raise ValueError("--min-pair-distance不能小于0")

    experiment = InteractiveRectangleExperiment(
        npz_path=args.map,
        config_path=args.config,
        n=args.n,
        seed=args.seed,
        obstacle_clearance=args.obstacle_clearance,
        boundary_clearance=args.boundary_clearance,
        min_point_spacing=args.min_point_spacing,
        min_pair_distance=args.min_pair_distance,
        output_dir=args.output_dir,
        save_plots=not args.no_plots
    )

    experiment.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
