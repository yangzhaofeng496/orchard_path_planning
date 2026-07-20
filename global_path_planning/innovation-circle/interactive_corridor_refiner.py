#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Corridor Path Refiner

Click to set start/goal, press Enter to plan,
drag two circles and press 'C' to refine the path.

Usage:
    python interactive_corridor_refiner.py

Controls:
    - Left click: Set start point (first click) and goal point (second click)
    - Press Enter: Plan path using RRT*
    - Drag black circles: Position two circles to define corridor
    - Press 'C': Refine path through the corridor
    - Press 'R': Reset (clear start/goal/path)
    - Press 'Q': Quit

Author: Claude Code
Date: 2026-07-19
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch

# 导入路径修正模块（优先使用增强版）
try:
    from circle_corridor_refiner_enhanced import refine_path_between_circles
    print("✓ Using enhanced corridor refiner (with connection & smoothing)")
    USE_ENHANCED = True
except ImportError:
    from circle_corridor_refiner import refine_path_between_circles
    print("✓ Using basic corridor refiner")
    USE_ENHANCED = False

import yaml
import os

# Import RRT* planner
sys.path.append('..')
try:
    from rrt_star import RRTStar
    HAS_RRT = True
except ImportError:
    print("Warning: RRT* planner not found, path planning disabled")
    HAS_RRT = False


class InteractiveCorridorRefiner:
    """Interactive corridor path refiner with draggable circles"""

    def __init__(self, config_file='corridor_config.yaml', preset='default'):
        """Initialize

        Args:
            config_file: Path to yaml configuration file
            preset: Preset name ('default', 'narrow', 'wide', 'tight', 'asymmetric', 'diagonal')
        """
        # Load configuration
        self.config = self._load_config(config_file)
        self.preset = preset

        self.obstacles = []  # No obstacles for now
        self.path = None  # Planned path
        self.refined_path = None  # Refined path
        self.circles = []  # Two draggable circles: [(x, y, r), (x, y, r)]
        self.start = None  # Start point
        self.goal = None  # Goal point
        self.fig = None
        self.ax = None

        # Visualization elements
        self.circle_patches = []
        self.circle_centers = []
        self.start_marker = None
        self.goal_marker = None
        self.path_line = None
        self.refined_path_line = None
        self.info_text = None

        # Dragging state
        self.dragging = False
        self.drag_circle_idx = None

        # Initialize circles from config
        self._init_circles_from_config()

    def _load_config(self, config_file):
        """Load configuration from yaml file"""
        if not os.path.exists(config_file):
            print(f"Warning: Config file '{config_file}' not found, using defaults")
            return self._get_default_config()

        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Configuration loaded from: {config_file}")
            return config
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            return self._get_default_config()

    def _get_default_config(self):
        """Get default configuration"""
        return {
            'visualization': {
                'figure_size': [14, 10],
                'axis_limits': {'x_min': -5, 'x_max': 25, 'y_min': -8, 'y_max': 8},
                'grid': True,
                'grid_alpha': 0.3
            },
            'circles': {
                'circle_1': {'x': 10.0, 'y': 3.0, 'radius': 0.8, 'color': 'black',
                            'linestyle': '--', 'linewidth': 3, 'alpha': 0.9},
                'circle_2': {'x': 10.0, 'y': -3.0, 'radius': 0.8, 'color': 'black',
                            'linestyle': '--', 'linewidth': 3, 'alpha': 0.9},
                'center': {'color': 'red', 'marker': 'o', 'markersize': 10, 'alpha': 0.9}
            },
            'path': {
                'num_points': 50,
                'original': {'color': 'blue', 'linestyle': '-', 'linewidth': 2,
                           'alpha': 0.6, 'label': 'Planned Path'},
                'refined': {'color': 'green', 'linestyle': '-', 'linewidth': 3,
                          'alpha': 0.9, 'label': 'Refined Path'}
            },
            'markers': {
                'start': {'color': 'green', 'marker': 'o', 'markersize': 15,
                         'label': 'Start', 'zorder': 10},
                'goal': {'color': 'red', 'marker': '*', 'markersize': 20,
                        'label': 'Goal', 'zorder': 10}
            },
            'presets': {
                'default': {
                    'circle_1': {'x': 10.0, 'y': 3.0, 'radius': 0.8},
                    'circle_2': {'x': 10.0, 'y': -3.0, 'radius': 0.8}
                }
            }
        }

    def _init_circles_from_config(self):
        """Initialize circles from configuration"""
        # Check if preset exists
        if self.preset in self.config.get('presets', {}):
            preset_config = self.config['presets'][self.preset]
            print(f"Using preset: {self.preset}")

            c1 = preset_config['circle_1']
            c2 = preset_config['circle_2']
            self.circles = [
                (c1['x'], c1['y'], c1['radius']),
                (c2['x'], c2['y'], c2['radius'])
            ]
        else:
            # Use default circle configuration
            c1 = self.config['circles']['circle_1']
            c2 = self.config['circles']['circle_2']
            self.circles = [
                (c1['x'], c1['y'], c1['radius']),
                (c2['x'], c2['y'], c2['radius'])
            ]
            print(f"Preset '{self.preset}' not found, using default configuration")

    def create_plot(self):
        """Create plot window"""
        # Get visualization settings
        vis_config = self.config['visualization']
        fig_size = vis_config['figure_size']
        axis_limits = vis_config['axis_limits']

        self.fig, self.ax = plt.subplots(figsize=tuple(fig_size))
        self.ax.set_aspect('equal')
        self.ax.grid(vis_config['grid'], alpha=vis_config['grid_alpha'])
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        self.ax.set_title('Interactive Corridor Refiner - Click start/goal, Enter to plan, Drag circles, C to refine, R to reset',
                         fontsize=12, fontweight='bold')

        # Set axis limits
        self.ax.set_xlim(axis_limits['x_min'], axis_limits['x_max'])
        self.ax.set_ylim(axis_limits['y_min'], axis_limits['y_max'])

        # Draw two draggable circles from config
        circle_config = self.config['circles']
        center_config = circle_config['center']

        self.circle_patches = []
        self.circle_centers = []

        for i, (x, y, r) in enumerate(self.circles):
            # Get style from config
            circle_key = f'circle_{i+1}'
            style = circle_config.get(circle_key, circle_config['circle_1'])

            # Circle outline
            circle = CirclePatch(
                (x, y), r,
                fill=False,
                edgecolor=style.get('color', 'black'),
                linewidth=style.get('linewidth', 3),
                linestyle=style.get('linestyle', '--'),
                alpha=style.get('alpha', 0.9),
                picker=self.config['interaction']['picker_tolerance']
            )
            self.ax.add_patch(circle)
            self.circle_patches.append(circle)

            # Circle center (draggable)
            center, = self.ax.plot(x, y, center_config['marker'],
                                  color=center_config['color'],
                                  markersize=center_config['markersize'],
                                  alpha=center_config['alpha'],
                                  picker=self.config['interaction']['picker_tolerance'])
            self.circle_centers.append(center)

        # Info text
        info_config = self.config['info_box']
        self.info_text = self.ax.text(
            info_config['position'][0], info_config['position'][1],
            self._get_info_text(),
            transform=self.ax.transAxes,
            verticalalignment=info_config['verticalalignment'],
            fontsize=info_config['fontsize'],
            bbox=info_config['bbox']
        )

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def _get_info_text(self):
        """Get info text"""
        if self.start is None:
            return "Step 1: Click to set START point"
        elif self.goal is None:
            return "Step 2: Click to set GOAL point"
        elif self.path is None:
            return "Step 3: Press ENTER to plan path"
        else:
            info = f"Path planned ({len(self.path)} points)\n"
            info += f"Circle 1: ({self.circles[0][0]:.1f}, {self.circles[0][1]:.1f}), r={self.circles[0][2]:.1f}\n"
            info += f"Circle 2: ({self.circles[1][0]:.1f}, {self.circles[1][1]:.1f}), r={self.circles[1][2]:.1f}\n"
            D = np.hypot(self.circles[1][0] - self.circles[0][0],
                         self.circles[1][1] - self.circles[0][1])
            gap = D - self.circles[0][2] - self.circles[1][2]
            info += f"Gap: {gap:.2f}m\n"
            info += "Drag circles, press C to refine, R to reset"
            return info

    def on_press(self, event):
        """Mouse press event"""
        if event.inaxes != self.ax:
            return

        # Check if clicking on a circle center (for dragging)
        for i, center in enumerate(self.circle_centers):
            contains, _ = center.contains(event)
            if contains:
                self.dragging = True
                self.drag_circle_idx = i
                return

        # If not dragging, handle start/goal selection
        if not self.dragging and event.button == 1:  # Left click
            if self.start is None:
                # Set start point
                self.start = (event.xdata, event.ydata)
                if self.start_marker is not None:
                    self.start_marker.remove()

                # Get marker config
                marker_config = self.config['markers']['start']
                self.start_marker, = self.ax.plot(
                    self.start[0], self.start[1],
                    marker_config['marker'],
                    color=marker_config['color'],
                    markersize=marker_config['markersize'],
                    label=marker_config['label'],
                    zorder=marker_config['zorder']
                )
                print(f"Start set: ({self.start[0]:.2f}, {self.start[1]:.2f})")
                self.info_text.set_text(self._get_info_text())
                self._update_legend()
                self.fig.canvas.draw()

            elif self.goal is None:
                # Set goal point
                self.goal = (event.xdata, event.ydata)
                if self.goal_marker is not None:
                    self.goal_marker.remove()

                # Get marker config
                marker_config = self.config['markers']['goal']
                self.goal_marker, = self.ax.plot(
                    self.goal[0], self.goal[1],
                    marker_config['marker'],
                    color=marker_config['color'],
                    markersize=marker_config['markersize'],
                    label=marker_config['label'],
                    zorder=marker_config['zorder']
                )
                print(f"Goal set: ({self.goal[0]:.2f}, {self.goal[1]:.2f})")
                print("Press ENTER to plan path")
                self.info_text.set_text(self._get_info_text())
                self._update_legend()
                self.fig.canvas.draw()

    def on_release(self, event):
        """Mouse release event"""
        self.dragging = False
        self.drag_circle_idx = None

    def on_motion(self, event):
        """Mouse motion event"""
        if not self.dragging or event.inaxes != self.ax:
            return

        if self.drag_circle_idx is None:
            return

        # Update circle position
        new_x, new_y = event.xdata, event.ydata
        idx = self.drag_circle_idx
        x, y, r = self.circles[idx]

        # Update circle data
        self.circles[idx] = (new_x, new_y, r)

        # Update visualization
        self.circle_patches[idx].center = (new_x, new_y)
        self.circle_centers[idx].set_data([new_x], [new_y])

        # Update info text
        if self.path is not None:
            self.info_text.set_text(self._get_info_text())

        # Clear refined path when circles move
        if self.refined_path_line is not None:
            self.refined_path_line.remove()
            self.refined_path_line = None
            self._update_legend()

        self.fig.canvas.draw()

    def on_key(self, event):
        """Key press event"""
        if event.key == 'return' or event.key == 'enter':
            # Plan path with RRT*
            if self.start is not None and self.goal is not None:
                self._plan_path()
            else:
                print("Please set start and goal points first")

        elif event.key == 'c' or event.key == 'C':
            # Refine path
            if self.path is not None:
                self._refine_path()
            else:
                print("Please plan path first (press Enter)")

        elif event.key == 'r' or event.key == 'R':
            # Reset
            self._reset()

        elif event.key == 'q' or event.key == 'Q':
            # Quit
            print("Quit")
            plt.close(self.fig)

    def _plan_path(self):
        """Plan path using simple straight line (no obstacles)"""
        print("\n" + "="*60)
        print("Planning path...")
        print(f"  Start: ({self.start[0]:.2f}, {self.start[1]:.2f})")
        print(f"  Goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f})")

        # Create simple straight line path
        N = self.config['path']['num_points']
        x = np.linspace(self.start[0], self.goal[0], N)
        y = np.linspace(self.start[1], self.goal[1], N)
        yaw = np.arctan2(self.goal[1] - self.start[1],
                         self.goal[0] - self.start[0]) * np.ones(N)
        self.path = np.column_stack([x, y, yaw])

        print(f"Path planned successfully! {len(self.path)} points")
        print("="*60 + "\n")

        # Draw path
        if self.path_line is not None:
            self.path_line.remove()

        # Get path style from config
        path_style = self.config['path']['original']
        self.path_line, = self.ax.plot(
            self.path[:, 0], self.path[:, 1],
            color=path_style['color'],
            linestyle=path_style['linestyle'],
            linewidth=path_style['linewidth'],
            alpha=path_style['alpha'],
            label=path_style['label']
        )

        # Update info and legend
        self.info_text.set_text(self._get_info_text())
        self._update_legend()
        self.fig.canvas.draw()

    def _reset(self):
        """Reset all"""
        print("\n" + "="*60)
        print("Resetting...")
        print("="*60 + "\n")

        # Clear start/goal
        self.start = None
        self.goal = None
        self.path = None
        self.refined_path = None

        # Remove markers
        if self.start_marker is not None:
            self.start_marker.remove()
            self.start_marker = None
        if self.goal_marker is not None:
            self.goal_marker.remove()
            self.goal_marker = None

        # Remove paths
        if self.path_line is not None:
            self.path_line.remove()
            self.path_line = None
        if self.refined_path_line is not None:
            self.refined_path_line.remove()
            self.refined_path_line = None

        # Update info text and legend
        self.info_text.set_text(self._get_info_text())
        self._update_legend()
        self.fig.canvas.draw()

    def _update_legend(self):
        """Update legend"""
        handles = []
        labels = []

        if self.start_marker:
            handles.append(self.start_marker)
            labels.append('Start')
        if self.goal_marker:
            handles.append(self.goal_marker)
            labels.append('Goal')
        if self.path_line:
            handles.append(self.path_line)
            labels.append('Planned Path')
        if self.refined_path_line:
            handles.append(self.refined_path_line)
            labels.append('Refined Path')

        if len(handles) > 0:
            self.ax.legend(handles, labels, loc='upper right', fontsize=10)
        else:
            # Clear legend if no handles
            legend = self.ax.get_legend()
            if legend:
                legend.remove()

    def _refine_path(self):
        """Execute path refinement"""
        print("\n" + "="*60)
        print("Refining path...")

        circle1 = self.circles[0]
        circle2 = self.circles[1]

        print(f"Circle 1: center=({circle1[0]:.1f}, {circle1[1]:.1f}), r={circle1[2]:.1f}")
        print(f"Circle 2: center=({circle2[0]:.1f}, {circle2[1]:.1f}), r={circle2[2]:.1f}")

        # Call refine function
        refined_path, info = refine_path_between_circles(
            self.path, circle1, circle2
        )

        if info['valid']:
            self.refined_path = refined_path
            print("\nRefinement successful!")
            print(f"  Corridor center: ({info['center_point'][0]:.3f}, {info['center_point'][1]:.3f})")
            print(f"  Target yaw: {np.degrees(info['target_yaw']):.1f}°")
            print(f"  Free gap: {info['free_gap']:.3f}m")
            print(f"  Center error improvement: {info['center_error_reduction']:.1f}% "
                  f"({info['original_center_error']:.3f}m → {info['refined_center_error']:.3f}m)")
            print(f"  Angle error improvement: {info['angle_error_reduction']:.1f}% "
                  f"({info['original_angle_error']:.1f}° → {info['refined_angle_error']:.1f}°)")

            # Draw refined path
            self._draw_refined_path(info)

        else:
            print(f"\nRefinement failed: {info.get('reason', 'Unknown')}")

        print("="*60 + "\n")

    def _draw_refined_path(self, info):
        """Draw refined path"""
        # Remove old refined path
        if self.refined_path_line is not None:
            self.refined_path_line.remove()

        # Get refined path style from config
        refined_style = self.config['path']['refined']
        self.refined_path_line, = self.ax.plot(
            self.refined_path[:, 0], self.refined_path[:, 1],
            color=refined_style['color'],
            linestyle=refined_style['linestyle'],
            linewidth=refined_style['linewidth'],
            alpha=refined_style['alpha'],
            label=refined_style['label']
        )

        # Mark corridor center
        Q = info['center_point']
        corridor_marker = self.config['markers']['corridor_center']
        self.ax.plot(Q[0], Q[1],
                    corridor_marker['marker'],
                    color=corridor_marker['color'],
                    markersize=corridor_marker['markersize'],
                    label=corridor_marker['label'],
                    zorder=corridor_marker['zorder'])

        # Mark adjustment range
        start_idx = info['start_index']
        end_idx = info['end_index']

        adjust_start_marker = self.config['markers']['adjust_start']
        self.ax.plot(self.refined_path[start_idx, 0], self.refined_path[start_idx, 1],
                    adjust_start_marker['marker'],
                    color=adjust_start_marker['color'],
                    markersize=adjust_start_marker['markersize'],
                    label=adjust_start_marker['label'],
                    zorder=adjust_start_marker['zorder'])

        adjust_end_marker = self.config['markers']['adjust_end']
        self.ax.plot(self.refined_path[end_idx, 0], self.refined_path[end_idx, 1],
                    adjust_end_marker['marker'],
                    color=adjust_end_marker['color'],
                    markersize=adjust_end_marker['markersize'],
                    label=adjust_end_marker['label'],
                    zorder=adjust_end_marker['zorder'])

        # Update legend
        self._update_legend()

        # Update info text
        info_str = f"Refinement completed!\n"
        info_str += f"Center error: {info['original_center_error']:.3f}m → {info['refined_center_error']:.3f}m\n"
        info_str += f"Improvement: {info['center_error_reduction']:.1f}%"
        self.info_text.set_text(info_str)

        self.fig.canvas.draw()

    def run(self):
        """Run interactive interface"""
        self.create_plot()
        plt.show()


# ============================================================================
# Main Program
# ============================================================================

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Interactive Corridor Path Refiner')
    parser.add_argument('--config', type=str, default='corridor_config.yaml',
                       help='Path to yaml configuration file (default: corridor_config.yaml)')
    parser.add_argument('--preset', type=str, default='default',
                       choices=['default', 'narrow', 'wide', 'tight', 'asymmetric', 'diagonal'],
                       help='Preset configuration to use (default: default)')
    args = parser.parse_args()

    print("="*80)
    print("Interactive Corridor Path Refiner")
    print("="*80)
    print(f"\nConfiguration file: {args.config}")
    print(f"Preset: {args.preset}")
    print("\nControls:")
    print("  1. Left click to set START point (green circle)")
    print("  2. Left click to set GOAL point (red star)")
    print("  3. Press ENTER to plan path (straight line)")
    print("  4. Drag red circle centers to position the two circles")
    print("  5. Press 'C' to refine path through the corridor")
    print("  6. Press 'R' to reset (clear start/goal/path)")
    print("  7. Press 'Q' to quit")
    print("\nAvailable presets:")
    print("  - default:    Standard corridor (r=0.8m, gap=4.4m)")
    print("  - narrow:     Narrow corridor (r=0.6m, gap=2.8m)")
    print("  - wide:       Wide corridor (r=1.0m, gap=6.0m)")
    print("  - tight:      Tight corridor (r=0.5m, gap=2.0m)")
    print("  - asymmetric: Different circle sizes (r1=1.2m, r2=0.6m)")
    print("  - diagonal:   Circles at different x positions")
    print("\n" + "="*80 + "\n")

    # Create interactive refiner
    refiner = InteractiveCorridorRefiner(config_file=args.config, preset=args.preset)
    refiner.run()

    print("\nProgram ended")


if __name__ == "__main__":
    main()
