"""
TEB 交互式演示
- 左键拖动障碍物
- 右键添加新障碍物
- 左键拖动起点/终点调整位置
- R 键重置场景
- 空格键暂停/继续优化
"""

import sys
import os
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from matplotlib.patches import Circle, FancyArrow, Polygon
from matplotlib import font_manager

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from local_path_planning import load_teb_config, TEBPlanner
from local_path_planning import VehicleState, Pose, CircleObstacle

# Matplotlib 中文字体配置
font_candidates = [
    "PingFang SC",       # macOS
    "Heiti SC",          # macOS
    "Songti SC",         # macOS
    "Microsoft YaHei",   # Windows
    "SimHei",            # Windows
    "Noto Sans CJK SC",  # Linux
    "WenQuanYi Micro Hei",
]

available_fonts = {font.name for font in font_manager.fontManager.ttflist}

for font_name in font_candidates:
    if font_name in available_fonts:
        plt.rcParams["font.sans-serif"] = [font_name]
        break

plt.rcParams["axes.unicode_minus"] = False


class InteractiveTEBTest:
    """交互式 TEB 演示"""

    def __init__(self):
        # 交互程序与主 TEB 共用同一份配置，避免参数和求解器设置漂移。
        config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'configs', 'teb_config.yaml'
        ))
        self.config = load_teb_config(config_path)
        print(f"[配置] 使用配置文件: {config_path}")

        # 设置边界
        self.bounds = (0.0, 20.0, 0.0, 12.0)

        # 创建 TEB 规划器
        self.planner = TEBPlanner(self.config, self.bounds)

        # 初始化起点和终点
        self.start_pos = [2.0, 6.0, 0.0]  # [x, y, yaw]
        self.goal_pos = [18.0, 6.0, 0.0]   # [x, y, yaw]

        # 初始化障碍物
        self.obstacles = [
            CircleObstacle(10.0, 6.0, 1.5),
        ]

        # 交互状态
        self.dragged_obstacle_index = None
        self.dragged_point = None  # 'start' 或 'goal'
        self.paused = False
        self.planning_enabled = True

        # TEB 规划结果
        self.teb_trajectory = []
        self.teb_nodes = []
        self.last_result = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='teb-planner')
        self.planning_future = None
        self.replan_requested = False
        self.planning = False

        # 创建图形界面
        self.fig, self.ax = plt.subplots(figsize=(14, 9))

        # 绘制静态元素
        self.draw_static_scene()

        # 动态元素
        self.start_patch = None
        self.goal_patch = None
        self.start_arrow = None
        self.goal_arrow = None
        self.obstacle_patches = []
        self.trajectory_line = None
        self.node_scatter = None
        self.vehicle_patches = []
        self.info_text_obj = None

        self.refresh_all_patches()

        # 连接事件
        self.connect_events()

        # 定时器：定期重新规划
        # GUI 线程只负责轮询优化结果，耗时的 SLSQP 在后台线程执行。
        # 增加间隔到 100ms 减少 CPU 占用
        self.timer = self.fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self.update)
        self.timer.start()

        # 首次规划
        self.replan()



    def draw_static_scene(self):
        """绘制静态场景元素"""
        x_min, x_max, y_min, y_max = self.bounds

        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X / m", fontsize=12)
        self.ax.set_ylabel("Y / m", fontsize=12)

        title_text = (
            "TEB 交互式演示 | "
            "左键拖动: 起点/终点/障碍物 | "
            "右键: 添加障碍物 | "
            "空格: 暂停/继续 | "
            "R: 重置"
        )
        self.ax.set_title(title_text, fontsize=11)

    def refresh_all_patches(self):
        """刷新所有可视化元素"""
        self.refresh_points()
        self.refresh_obstacles()
        self.refresh_trajectory()
        self.refresh_info_text()

    def refresh_points(self):
        """刷新起点和终点"""
        # 移除旧的
        if self.start_patch:
            self.start_patch.remove()
        if self.goal_patch:
            self.goal_patch.remove()
        if self.start_arrow:
            self.start_arrow.remove()
        if self.goal_arrow:
            self.goal_arrow.remove()

        # 起点
        self.start_patch = Circle(
            (self.start_pos[0], self.start_pos[1]),
            0.4,
            facecolor='green',
            edgecolor='darkgreen',
            linewidth=2,
            alpha=0.7,
            label='起点 (可拖动)',
            zorder=10,
        )
        self.ax.add_patch(self.start_patch)

        # 起点朝向箭头
        arrow_len = 0.8
        dx = arrow_len * math.cos(self.start_pos[2])
        dy = arrow_len * math.sin(self.start_pos[2])
        self.start_arrow = FancyArrow(
            self.start_pos[0], self.start_pos[1], dx, dy,
            width=0.15, head_width=0.4, head_length=0.3,
            fc='darkgreen', ec='darkgreen', zorder=11,
        )
        self.ax.add_patch(self.start_arrow)

        # 终点
        self.goal_patch = Circle(
            (self.goal_pos[0], self.goal_pos[1]),
            0.4,
            facecolor='red',
            edgecolor='darkred',
            linewidth=2,
            alpha=0.7,
            label='终点 (可拖动)',
            zorder=10,
        )
        self.ax.add_patch(self.goal_patch)

        # 终点朝向箭头
        dx = arrow_len * math.cos(self.goal_pos[2])
        dy = arrow_len * math.sin(self.goal_pos[2])
        self.goal_arrow = FancyArrow(
            self.goal_pos[0], self.goal_pos[1], dx, dy,
            width=0.15, head_width=0.4, head_length=0.3,
            fc='darkred', ec='darkred', zorder=11,
        )
        self.ax.add_patch(self.goal_arrow)

    def refresh_obstacles(self):
        """刷新障碍物"""
        for patch in self.obstacle_patches:
            patch.remove()
        self.obstacle_patches = []

        for i, obs in enumerate(self.obstacles):
            patch = Circle(
                (obs.x, obs.y),
                obs.radius,
                facecolor='orange',
                edgecolor='darkorange',
                linewidth=2,
                alpha=0.6,
                label='障碍物 (可拖动)' if i == 0 else None,
                zorder=5,
            )
            self.ax.add_patch(patch)
            self.obstacle_patches.append(patch)

    def refresh_trajectory(self):
        """刷新 TEB 轨迹"""
        # 移除旧的轨迹线
        if self.trajectory_line:
            self.trajectory_line.remove()
            self.trajectory_line = None

        # 移除旧的节点散点
        if self.node_scatter:
            self.node_scatter.remove()
            self.node_scatter = None

        # 移除旧的车辆轮廓
        for patch in self.vehicle_patches:
            patch.remove()
        self.vehicle_patches = []

        if not self.teb_trajectory:
            return

        # 绘制轨迹线
        traj_x = [p.x for p in self.teb_trajectory]
        traj_y = [p.y for p in self.teb_trajectory]
        self.trajectory_line, = self.ax.plot(
            traj_x, traj_y,
            'b-',
            linewidth=2,
            alpha=0.6,
            label='TEB 优化轨迹',
            zorder=3,
        )

        # 绘制 TEB 节点
        self.node_scatter = self.ax.scatter(
            traj_x, traj_y,
            s=40,
            c='blue',
            marker='o',
            alpha=0.5,
            zorder=4,
        )

        # 计算每个节点的速度和绘制方向箭头
        teb_nodes = self.planner.teb_nodes if hasattr(self.planner, 'teb_nodes') and self.planner.teb_nodes else []

        for i in range(len(self.teb_trajectory)):
            pose = self.teb_trajectory[i]

            # 绘制方向箭头
            arrow_length = 0.5
            dx = arrow_length * math.cos(pose.yaw)
            dy = arrow_length * math.sin(pose.yaw)

            arrow = FancyArrow(
                pose.x, pose.y, dx, dy,
                width=0.08,
                head_width=0.25,
                head_length=0.2,
                fc='darkblue',
                ec='darkblue',
                alpha=0.7,
                zorder=5,
            )
            self.ax.add_patch(arrow)
            self.vehicle_patches.append(arrow)

            # 计算并显示速度
            if i < len(teb_nodes) and i < len(teb_nodes) - 1:
                # 从 TEB 节点计算速度
                node_curr = teb_nodes[i]
                node_next = teb_nodes[i + 1]
                dist = math.hypot(node_next.x - node_curr.x, node_next.y - node_curr.y)
                speed = dist / node_curr.dt if node_curr.dt > 0.001 else 0.0

                # 在节点旁边显示速度文本
                speed_text = self.ax.text(
                    pose.x + 0.3, pose.y + 0.3,
                    f'{speed:.1f}',
                    fontsize=8,
                    color='darkblue',
                    alpha=0.8,
                    zorder=6,
                )
                self.vehicle_patches.append(speed_text)

        # 绘制部分车辆轮廓（每隔几个节点）
        step = max(1, len(self.teb_trajectory) // 5)
        for i in range(0, len(self.teb_trajectory), step):
            pose = self.teb_trajectory[i]
            corners = self.get_vehicle_corners(pose.x, pose.y, pose.yaw)
            vehicle_patch = Polygon(
                corners,
                closed=True,
                fill=False,
                edgecolor='blue',
                linewidth=1.5,
                alpha=0.4,
                zorder=2,
            )
            self.ax.add_patch(vehicle_patch)
            self.vehicle_patches.append(vehicle_patch)

    def get_vehicle_corners(self, x, y, yaw):
        """获取车辆矩形的四个角点"""
        front = self.config.vehicle_front_length
        rear = self.config.vehicle_rear_length
        half_width = self.config.vehicle_width / 2.0

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        local_corners = [
            (front, half_width),
            (front, -half_width),
            (-rear, -half_width),
            (-rear, half_width),
        ]

        corners = []
        for local_x, local_y in local_corners:
            world_x = x + local_x * cos_yaw - local_y * sin_yaw
            world_y = y + local_x * sin_yaw + local_y * cos_yaw
            corners.append((world_x, world_y))

        return corners

    def refresh_info_text(self):
        """刷新信息文本"""
        if self.info_text_obj:
            self.info_text_obj.remove()

        if self.last_result and self.last_result.success:
            status = "[成功] 规划成功"
            color = 'lightgreen'
        elif self.planning:
            status = "[规划中] 正在更新轨迹..."
            color = 'lightyellow'
        elif self.last_result:
            reason = getattr(self.planner, 'last_failure_reason', '未知')
            status = f"[失败] 规划失败: {reason}"
            color = 'lightsalmon'
        else:
            status = "等待规划..."
            color = 'lightgray'

        info_lines = [
            f"状态: {status}",
            f"",
            f"起点: ({self.start_pos[0]:.1f}, {self.start_pos[1]:.1f})",
            f"终点: ({self.goal_pos[0]:.1f}, {self.goal_pos[1]:.1f})",
            f"障碍物数量: {len(self.obstacles)}",
            f"TEB 节点数: {len(self.teb_trajectory)}",
            f"",
            f"暂停: {'是' if self.paused else '否'}",
        ]

        info_text = '\n'.join(info_lines)

        self.info_text_obj = self.ax.text(
            0.02, 0.98, info_text,
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8),
            zorder=100,
        )

    def replan(self):
        """提交异步重规划；繁忙时只保留最新场景。"""
        if not self.planning_enabled or self.paused:
            return

        if self.planning_future is not None and not self.planning_future.done():
            self.replan_requested = True
            return

        start_pos = tuple(self.start_pos)
        goal_pos = tuple(self.goal_pos)
        obstacles = [CircleObstacle(obs.x, obs.y, obs.radius) for obs in self.obstacles]
        config = copy.deepcopy(self.config)
        self.planning = True
        self.replan_requested = False
        self.refresh_info_text()
        self.fig.canvas.draw_idle()
        self.planning_future = self.executor.submit(
            self._planning_job, config, self.bounds, start_pos, goal_pos, obstacles
        )

    @staticmethod
    def _planning_job(config, bounds, start_pos, goal_pos, obstacles):
        """后台规划任务，不接触任何 Matplotlib 对象。"""
        # 交互演示需要展示一条真正到达红色终点的完整轨迹；核心规划器
        # 在车辆实时运行时仍使用配置中的局部前瞻窗口。
        start_goal_distance = math.hypot(
            goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1]
        )
        config.lookahead_distance = max(config.lookahead_distance, start_goal_distance + 1e-6)
        num_points = 10
        global_path = []
        for i in range(num_points):
            ratio = i / (num_points - 1)
            x = start_pos[0] + ratio * (goal_pos[0] - start_pos[0])
            y = start_pos[1] + ratio * (goal_pos[1] - start_pos[1])
            yaw = math.atan2(
                goal_pos[1] - start_pos[1],
                goal_pos[0] - start_pos[0]
            )
            global_path.append(Pose(x, y, yaw))

        planner = TEBPlanner(config, bounds)
        planner.set_global_path(global_path)
        current_state = VehicleState(
            x=start_pos[0],
            y=start_pos[1],
            yaw=start_pos[2],
            speed=0.5,
            steering=0.0,
        )
        result = planner.plan(current_state, obstacles)
        return planner, result

    def update(self):
        """在 GUI 线程接收后台结果，并立即处理拖动期间的最新请求。"""
        if self.planning_future is None or not self.planning_future.done():
            return
        try:
            self.planner, self.last_result = self.planning_future.result()
            if self.last_result and self.last_result.success:
                self.teb_trajectory = self.last_result.trajectory
            else:
                self.teb_trajectory = []
        except Exception as exc:
            self.last_result = None
            self.teb_trajectory = []
            self.planner.last_failure_reason = f"后台规划异常: {exc}"
        finally:
            self.planning_future = None
            self.planning = False

        self.refresh_trajectory()
        self.refresh_info_text()
        self.fig.canvas.draw_idle()
        if self.replan_requested and not self.paused:
            self.replan()

    def connect_events(self):
        """连接鼠标和键盘事件"""
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('close_event', self.on_close)

    def on_mouse_press(self, event):
        """鼠标按下事件"""
        if event.inaxes != self.ax:
            return

        # 右键：添加障碍物
        if event.button == 3:
            self.obstacles.append(CircleObstacle(event.xdata, event.ydata, 1.0))
            self.refresh_obstacles()
            self.replan()
            return

        # 左键：检查是否点击了起点、终点或障碍物
        if event.button == 1:
            # 检查起点
            dist_start = math.hypot(event.xdata - self.start_pos[0], event.ydata - self.start_pos[1])
            if dist_start <= 0.6:
                self.dragged_point = 'start'
                return

            # 检查终点
            dist_goal = math.hypot(event.xdata - self.goal_pos[0], event.ydata - self.goal_pos[1])
            if dist_goal <= 0.6:
                self.dragged_point = 'goal'
                return

            # 检查障碍物
            for i, obs in enumerate(self.obstacles):
                dist = math.hypot(event.xdata - obs.x, event.ydata - obs.y)
                if dist <= obs.radius + 0.3:
                    self.dragged_obstacle_index = i
                    return

    def on_mouse_motion(self, event):
        """鼠标移动事件"""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        x_min, x_max, y_min, y_max = self.bounds

        # 拖动起点
        if self.dragged_point == 'start':
            self.start_pos[0] = np.clip(event.xdata, x_min + 1, x_max - 1)
            self.start_pos[1] = np.clip(event.ydata, y_min + 1, y_max - 1)
            self.refresh_points()
            self.replan()
            return

        # 拖动终点
        if self.dragged_point == 'goal':
            self.goal_pos[0] = np.clip(event.xdata, x_min + 1, x_max - 1)
            self.goal_pos[1] = np.clip(event.ydata, y_min + 1, y_max - 1)
            self.refresh_points()
            self.replan()
            return

        # 拖动障碍物
        if self.dragged_obstacle_index is not None:
            obs = self.obstacles[self.dragged_obstacle_index]
            obs.x = np.clip(event.xdata, x_min + obs.radius, x_max - obs.radius)
            obs.y = np.clip(event.ydata, y_min + obs.radius, y_max - obs.radius)

            # 更新障碍物位置
            self.obstacle_patches[self.dragged_obstacle_index].center = (obs.x, obs.y)
            self.replan()
            self.fig.canvas.draw_idle()
            return

    def on_mouse_release(self, event):
        """鼠标释放事件"""
        self.dragged_obstacle_index = None
        self.dragged_point = None

    def on_key_press(self, event):
        """键盘按下事件"""
        if event.key == ' ':
            # 空格：暂停/继续
            self.paused = not self.paused
            self.refresh_info_text()
            self.fig.canvas.draw_idle()

        elif event.key in ('r', 'R'):
            # R：重置场景
            self.reset_scene()

        elif event.key == 'escape':
            # Esc：退出
            self.timer.stop()
            self.executor.shutdown(wait=False, cancel_futures=True)
            plt.close(self.fig)

    def on_close(self, event):
        """关闭窗口时停止后台线程。"""
        self.timer.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def reset_scene(self):
        """重置场景"""
        self.start_pos = [2.0, 6.0, 0.0]
        self.goal_pos = [18.0, 6.0, 0.0]
        self.obstacles = [CircleObstacle(10.0, 6.0, 1.5)]
        self.paused = False

        self.refresh_all_patches()
        self.replan()

        # 重新绘制图例
        self.ax.legend(loc='upper right', fontsize=9)

    def show(self):
        """显示窗口"""
        self.ax.legend(loc='upper right', fontsize=9)
        plt.tight_layout()
        plt.show()


def main():
    print("=" * 70)
    print("TEB 交互式演示")
    print("=" * 70)
    print("操作说明:")
    print("  左键拖动: 移动起点、终点或障碍物")
    print("  右键单击: 添加新障碍物")
    print("  空格键:   暂停/继续优化")
    print("  R 键:     重置场景")
    print("  Esc 键:   退出程序")
    print("=" * 70)

    app = InteractiveTEBTest()
    app.show()


if __name__ == "__main__":
    main()
