"""
TEB 动态调参工具
- 使用滑动条实时调整所有 TEB 优化权重
- 自动重新规划并显示效果
- 可以保存调好的参数到配置文件
"""

import sys
import os
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle, FancyArrow, Polygon
from matplotlib import font_manager

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from local_path_planning import load_teb_config, TEBPlanner
from local_path_planning import VehicleState, Pose, CircleObstacle

# Matplotlib 中文字体配置
font_candidates = [
    "PingFang SC",
    "Heiti SC",
    "Songti SC",
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
]

available_fonts = {font.name for font in font_manager.fontManager.ttflist}

for font_name in font_candidates:
    if font_name in available_fonts:
        plt.rcParams["font.sans-serif"] = [font_name]
        break

plt.rcParams["axes.unicode_minus"] = False


class TEBTuningTool:
    """TEB 动态调参工具"""

    def __init__(self):
        # 加载配置
        config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'configs', 'teb_config.yaml'
        ))
        self.config = load_teb_config(config_path)
        self.config_path = config_path
        print(f"[配置] 使用配置文件: {config_path}")

        # 设置边界
        self.bounds = (0.0, 20.0, 0.0, 12.0)

        # 创建 TEB 规划器
        self.planner = TEBPlanner(self.config, self.bounds)

        # 场景设置
        self.start_pos = [2.0, 6.0, 0.0]
        self.goal_pos = [18.0, 6.0, 0.0]
        self.obstacles = [
            CircleObstacle(10.0, 6.0, 1.5),
        ]

        # 创建全局路径
        self.create_global_path()

        # 创建图形界面
        self.fig = plt.figure(figsize=(18, 10))

        # 左侧：场景显示
        self.ax_scene = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)

        # 右侧：参数滑动条
        slider_positions = [
            (0, 2, 1, 1),  # w_time
            (0, 2, 2, 1),  # w_obstacle
            (1, 2, 1, 1),  # w_kinematics
            (1, 2, 2, 1),  # w_acceleration
            (2, 2, 1, 1),  # w_omega
            (2, 2, 2, 1),  # w_path
        ]

        # 参数定义：(名称, 当前值, 最小值, 最大值, 步长)
        self.param_definitions = [
            ('w_time', self.config.w_time, 0.0, 5.0, 0.1),
            ('w_obstacle', self.config.w_obstacle, 0.0, 100.0, 1.0),
            ('w_kinematics', self.config.w_kinematics, 0.0, 50.0, 1.0),
            ('w_acceleration', self.config.w_acceleration, 0.0, 20.0, 0.5),
            ('w_omega', self.config.w_omega, 0.0, 30.0, 1.0),
            ('w_path', self.config.w_path, 0.0, 5.0, 0.1),
        ]

        # 额外的参数（第二行）
        self.param_definitions_extra = [
            ('w_velocity', self.config.w_velocity, 0.0, 50.0, 1.0),
            ('w_steering', self.config.w_steering, 0.0, 50.0, 1.0),
            ('num_samples', self.config.num_samples, 5, 50, 1),
            ('max_iterations', self.config.max_iterations, 10, 200, 5),
        ]

        self.sliders = {}
        self.create_sliders()

        # 创建按钮
        self.create_buttons()

        # 绘制场景
        self.draw_scene()

        # 轨迹可视化元素
        self.trajectory_line = None
        self.node_scatter = None
        self.vehicle_patches = []
        self.info_text_obj = None

        # 执行首次规划
        self.replan()

    def create_global_path(self):
        """创建简单的全局路径"""
        path = [
            Pose(self.start_pos[0], self.start_pos[1], self.start_pos[2]),
            Pose(self.goal_pos[0], self.goal_pos[1], self.goal_pos[2]),
        ]
        self.planner.set_global_path(path)

    def create_sliders(self):
        """创建参数滑动条"""
        # 主要参数（上半部分，2列）
        for idx, (name, initial, vmin, vmax, step) in enumerate(self.param_definitions):
            row = idx // 2
            col = idx % 2

            # 计算位置
            left = 0.68 + col * 0.15
            bottom = 0.75 - row * 0.12
            width = 0.12
            height = 0.02

            ax_slider = self.fig.add_axes([left, bottom, width, height])
            slider = Slider(
                ax_slider,
                name,
                vmin,
                vmax,
                valinit=initial,
                valstep=step,
                color='skyblue'
            )
            slider.on_changed(self.on_slider_changed)
            self.sliders[name] = slider

        # 额外参数（下半部分，2列）
        for idx, (name, initial, vmin, vmax, step) in enumerate(self.param_definitions_extra):
            row = idx // 2
            col = idx % 2

            left = 0.68 + col * 0.15
            bottom = 0.35 - row * 0.12
            width = 0.12
            height = 0.02

            ax_slider = self.fig.add_axes([left, bottom, width, height])
            slider = Slider(
                ax_slider,
                name,
                vmin,
                vmax,
                valinit=initial,
                valstep=step,
                color='lightgreen'
            )
            slider.on_changed(self.on_slider_changed)
            self.sliders[name] = slider

    def create_buttons(self):
        """创建控制按钮"""
        # 重置按钮
        ax_reset = self.fig.add_axes([0.70, 0.05, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, '重置参数', color='lightcoral')
        self.btn_reset.on_clicked(self.on_reset_clicked)

        # 保存按钮
        ax_save = self.fig.add_axes([0.80, 0.05, 0.08, 0.04])
        self.btn_save = Button(ax_save, '保存配置', color='lightgreen')
        self.btn_save.on_clicked(self.on_save_clicked)

    def draw_scene(self):
        """绘制场景"""
        x_min, x_max, y_min, y_max = self.bounds

        self.ax_scene.clear()
        self.ax_scene.set_xlim(x_min, x_max)
        self.ax_scene.set_ylim(y_min, y_max)
        self.ax_scene.set_aspect("equal")
        self.ax_scene.grid(True, alpha=0.3)
        self.ax_scene.set_xlabel("X / m", fontsize=12)
        self.ax_scene.set_ylabel("Y / m", fontsize=12)
        self.ax_scene.set_title("TEB 动态调参工具 - 实时规划效果", fontsize=13, fontweight='bold')

        # 绘制起点
        start_circle = Circle(
            (self.start_pos[0], self.start_pos[1]),
            0.4,
            facecolor='green',
            edgecolor='darkgreen',
            linewidth=2,
            alpha=0.7,
            label='起点',
            zorder=10,
        )
        self.ax_scene.add_patch(start_circle)

        # 起点朝向箭头
        arrow_len = 0.8
        dx = arrow_len * math.cos(self.start_pos[2])
        dy = arrow_len * math.sin(self.start_pos[2])
        start_arrow = FancyArrow(
            self.start_pos[0], self.start_pos[1], dx, dy,
            width=0.15, head_width=0.4, head_length=0.3,
            fc='darkgreen', ec='darkgreen', zorder=11,
        )
        self.ax_scene.add_patch(start_arrow)

        # 绘制终点
        goal_circle = Circle(
            (self.goal_pos[0], self.goal_pos[1]),
            0.4,
            facecolor='red',
            edgecolor='darkred',
            linewidth=2,
            alpha=0.7,
            label='终点',
            zorder=10,
        )
        self.ax_scene.add_patch(goal_circle)

        # 终点朝向箭头
        dx = arrow_len * math.cos(self.goal_pos[2])
        dy = arrow_len * math.sin(self.goal_pos[2])
        goal_arrow = FancyArrow(
            self.goal_pos[0], self.goal_pos[1], dx, dy,
            width=0.15, head_width=0.4, head_length=0.3,
            fc='darkred', ec='darkred', zorder=11,
        )
        self.ax_scene.add_patch(goal_arrow)

        # 绘制障碍物
        for obs in self.obstacles:
            obs_circle = Circle(
                (obs.x, obs.y),
                obs.radius,
                facecolor='orange',
                edgecolor='darkorange',
                linewidth=2,
                alpha=0.6,
                label='障碍物',
                zorder=5,
            )
            self.ax_scene.add_patch(obs_circle)

        self.ax_scene.legend(loc='upper left', fontsize=10)

    def on_slider_changed(self, val):
        """滑动条改变时的回调"""
        self.replan()

    def on_reset_clicked(self, event):
        """重置按钮回调"""
        # 重新加载配置
        self.config = load_teb_config(self.config_path)

        # 重置所有滑动条
        self.sliders['w_time'].set_val(self.config.w_time)
        self.sliders['w_obstacle'].set_val(self.config.w_obstacle)
        self.sliders['w_kinematics'].set_val(self.config.w_kinematics)
        self.sliders['w_acceleration'].set_val(self.config.w_acceleration)
        self.sliders['w_omega'].set_val(self.config.w_omega)
        self.sliders['w_path'].set_val(self.config.w_path)
        self.sliders['w_velocity'].set_val(self.config.w_velocity)
        self.sliders['w_steering'].set_val(self.config.w_steering)
        self.sliders['num_samples'].set_val(self.config.num_samples)
        self.sliders['max_iterations'].set_val(self.config.max_iterations)

        print("[重置] 参数已重置到配置文件默认值")

    def on_save_clicked(self, event):
        """保存按钮回调"""
        try:
            # 读取当前配置文件
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            # 更新权重参数
            config_data['teb']['w_time'] = float(self.sliders['w_time'].val)
            config_data['teb']['w_obstacle'] = float(self.sliders['w_obstacle'].val)
            config_data['teb']['w_kinematics'] = float(self.sliders['w_kinematics'].val)
            config_data['teb']['w_acceleration'] = float(self.sliders['w_acceleration'].val)
            config_data['teb']['w_omega'] = float(self.sliders['w_omega'].val)
            config_data['teb']['w_path'] = float(self.sliders['w_path'].val)
            config_data['teb']['w_velocity'] = float(self.sliders['w_velocity'].val)
            config_data['teb']['w_steering'] = float(self.sliders['w_steering'].val)
            config_data['teb']['num_samples'] = int(self.sliders['num_samples'].val)
            config_data['teb']['max_iterations'] = int(self.sliders['max_iterations'].val)

            # 保存到文件
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False)

            print(f"[保存] 参数已保存到: {self.config_path}")
            print("=" * 60)
            print(f"w_time={self.sliders['w_time'].val:.1f}, "
                  f"w_obstacle={self.sliders['w_obstacle'].val:.1f}, "
                  f"w_kinematics={self.sliders['w_kinematics'].val:.1f}")
            print(f"w_acceleration={self.sliders['w_acceleration'].val:.1f}, "
                  f"w_omega={self.sliders['w_omega'].val:.1f}, "
                  f"w_path={self.sliders['w_path'].val:.1f}")
            print(f"w_velocity={self.sliders['w_velocity'].val:.1f}, "
                  f"w_steering={self.sliders['w_steering'].val:.1f}")
            print(f"num_samples={int(self.sliders['num_samples'].val)}, "
                  f"max_iterations={int(self.sliders['max_iterations'].val)}")
            print("=" * 60)

        except Exception as e:
            print(f"[错误] 保存配置失败: {e}")

    def replan(self):
        """使用当前参数重新规划"""
        # 更新规划器参数
        self.planner.w_time = self.sliders['w_time'].val
        self.planner.w_obstacle = self.sliders['w_obstacle'].val
        self.planner.w_kinematics = self.sliders['w_kinematics'].val
        self.planner.w_acceleration = self.sliders['w_acceleration'].val
        self.planner.w_omega = self.sliders['w_omega'].val
        self.planner.w_path = self.sliders['w_path'].val
        self.planner.w_velocity = self.sliders['w_velocity'].val
        self.planner.w_steering = self.sliders['w_steering'].val

        # 更新节点数和迭代次数
        self.config.num_samples = int(self.sliders['num_samples'].val)
        self.config.max_iterations = int(self.sliders['max_iterations'].val)

        # 重新创建规划器以应用新的节点数和迭代次数
        self.planner = TEBPlanner(self.config, self.bounds)
        self.planner.w_time = self.sliders['w_time'].val
        self.planner.w_obstacle = self.sliders['w_obstacle'].val
        self.planner.w_kinematics = self.sliders['w_kinematics'].val
        self.planner.w_acceleration = self.sliders['w_acceleration'].val
        self.planner.w_omega = self.sliders['w_omega'].val
        self.planner.w_path = self.sliders['w_path'].val
        self.planner.w_velocity = self.sliders['w_velocity'].val
        self.planner.w_steering = self.sliders['w_steering'].val

        self.create_global_path()

        # 创建当前状态
        current_state = VehicleState(
            x=self.start_pos[0],
            y=self.start_pos[1],
            yaw=self.start_pos[2],
            speed=0.0,
            steering=0.0,
        )

        # 执行规划
        result = self.planner.plan(current_state, self.obstacles)

        # 清除旧的轨迹可视化
        if self.trajectory_line:
            self.trajectory_line.remove()
            self.trajectory_line = None
        if self.node_scatter:
            self.node_scatter.remove()
            self.node_scatter = None
        for patch in self.vehicle_patches:
            patch.remove()
        self.vehicle_patches = []
        if self.info_text_obj:
            self.info_text_obj.remove()
            self.info_text_obj = None

        # 绘制新的轨迹
        if result and result.success:
            trajectory = result.trajectory

            # 轨迹线
            traj_x = [p.x for p in trajectory]
            traj_y = [p.y for p in trajectory]
            self.trajectory_line, = self.ax_scene.plot(
                traj_x, traj_y,
                'b-',
                linewidth=2,
                alpha=0.6,
                zorder=3,
            )

            # TEB 节点
            self.node_scatter = self.ax_scene.scatter(
                traj_x, traj_y,
                s=40,
                c='blue',
                marker='o',
                alpha=0.8,
                zorder=4,
            )

            # 车辆轮廓（每隔几个节点绘制一个）
            step = max(1, len(trajectory) // 5)
            for i in range(0, len(trajectory), step):
                pose = trajectory[i]
                vehicle_patch = self.draw_vehicle(pose.x, pose.y, pose.yaw)
                self.vehicle_patches.append(vehicle_patch)

            # 信息文本
            info_text = (
                f"✅ 规划成功\n"
                f"节点数: {len(trajectory)}\n"
                f"速度: {result.control.speed:.2f} m/s\n"
                f"转向: {math.degrees(result.control.steering):.1f}°\n"
                f"代价: {result.cost:.2f}"
            )
            self.info_text_obj = self.ax_scene.text(
                0.02, 0.98,
                info_text,
                transform=self.ax_scene.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                fontsize=10,
            )
        else:
            # 规划失败
            failure_reason = self.planner.last_failure_reason or "未知原因"
            info_text = f"❌ 规划失败\n原因: {failure_reason}"
            self.info_text_obj = self.ax_scene.text(
                0.02, 0.98,
                info_text,
                transform=self.ax_scene.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                fontsize=10,
            )

        self.fig.canvas.draw_idle()

    def draw_vehicle(self, x, y, yaw):
        """绘制车辆轮廓"""
        # 车辆尺寸
        front = self.config.vehicle_front_length
        rear = self.config.vehicle_rear_length
        width = self.config.vehicle_width

        # 车辆角点（车辆坐标系）
        corners_local = np.array([
            [front, width / 2],
            [front, -width / 2],
            [-rear, -width / 2],
            [-rear, width / 2],
        ])

        # 旋转矩阵
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rotation = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw],
        ])

        # 转换到世界坐标系
        corners_world = corners_local @ rotation.T + np.array([x, y])

        # 绘制多边形
        patch = Polygon(
            corners_world,
            closed=True,
            facecolor='cyan',
            edgecolor='blue',
            linewidth=1,
            alpha=0.3,
            zorder=6,
        )
        self.ax_scene.add_patch(patch)
        return patch

    def show(self):
        """显示界面"""
        plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("TEB 动态调参工具")
    print("=" * 60)
    print("功能:")
    print("  - 使用滑动条实时调整 TEB 优化权重")
    print("  - 立即重新规划并显示效果")
    print("  - 点击「保存配置」按钮将参数保存到配置文件")
    print("  - 点击「重置参数」按钮恢复配置文件默认值")
    print("=" * 60)

    tool = TEBTuningTool()
    tool.show()
