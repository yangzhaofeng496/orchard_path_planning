#!/usr/bin/env python3
"""Interactive circular-obstacle scene editor for the orchard planner.

The saved NPZ layout is compatible with the scene files used by this project:
``obstacles`` contains rows of ``(x, y, radius)`` and ``bounds`` is ordered as
``(xmin, xmax, ymin, ymax)``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, Slider
import numpy as np


# Prefer fonts commonly available on macOS/Linux so Chinese UI text renders.
mpl.rcParams["font.sans-serif"] = [
    "PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


@dataclass
class Scene:
    bounds: np.ndarray
    start_pos: np.ndarray
    goal_pos: np.ndarray
    rectangle: np.ndarray = field(
        default_factory=lambda: np.asarray([30.0, 22.0, 0.0], dtype=float)
    )
    obstacles: list[list[float]] = field(default_factory=list)
    corridors: np.ndarray = field(
        default_factory=lambda: np.empty((0, 5), dtype=float)
    )
    description: str = "交互式圆形障碍物地图"


def load_scene(path: Path) -> Scene:
    with np.load(path, allow_pickle=False) as data:
        bounds = np.asarray(data["bounds"], dtype=float)
        if bounds.shape != (4,):
            raise ValueError("bounds必须是[xmin, xmax, ymin, ymax]")
        obstacles = np.asarray(data["obstacles"], dtype=float).reshape((-1, 3))
        start_pos = np.asarray(data["start_pos"], dtype=float)
        goal_pos = np.asarray(data["goal_pos"], dtype=float)
        rectangle = np.asarray(
            data["rectangle"] if "rectangle" in data else [30.0, 22.0, 0.0],
            dtype=float,
        )
        corridors = np.asarray(
            data["corridors"] if "corridors" in data else np.empty((0, 5)),
            dtype=float,
        ).reshape((-1, 5))
        description = str(data["description"]) if "description" in data else path.stem

    return Scene(
        bounds=bounds,
        start_pos=start_pos,
        goal_pos=goal_pos,
        rectangle=rectangle,
        obstacles=obstacles.tolist(),
        corridors=corridors,
        description=description,
    )


def save_scene(path: Path, scene: Scene, resolution: float) -> None:
    path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    xmin, xmax, ymin, ymax = scene.bounds
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))
    obstacles = np.asarray(scene.obstacles, dtype=float).reshape((-1, 3))

    np.savez(
        path,
        format_version=np.asarray([2], dtype=np.int32),
        obstacles=obstacles,
        corridors=np.asarray(scene.corridors, dtype=float).reshape((-1, 5)),
        start_pos=np.asarray(scene.start_pos, dtype=float),
        goal_pos=np.asarray(scene.goal_pos, dtype=float),
        bounds=np.asarray(scene.bounds, dtype=float),
        rectangle=np.asarray(scene.rectangle, dtype=float),
        description=np.asarray(scene.description),
        map_resolution=np.asarray(resolution, dtype=float),
        map_size=np.asarray([width, height], dtype=np.int32),
        map_origin=np.asarray([xmin, ymin, 0.0], dtype=float),
    )
    print(f"[保存] {path}，圆形障碍物 {len(obstacles)} 个")


class CircleMapEditor:
    def __init__(self, scene: Scene, output: Path, radius: float, resolution: float):
        self.scene = scene
        self.output = output.with_suffix(".npz")
        self.default_radius = float(radius)
        self.resolution = float(resolution)
        self.selected: Optional[int] = None
        self.dragging = False
        self.click_offset = np.zeros(2, dtype=float)
        self.place_mode = "obstacle"
        self.history: list[tuple[list[list[float]], np.ndarray, np.ndarray]] = []

        self.fig, self.ax = plt.subplots(figsize=(9.5, 8.0))
        self.fig.subplots_adjust(bottom=0.20, right=0.82)
        self.fig.canvas.manager.set_window_title("圆形障碍物地图编辑器")

        slider_ax = self.fig.add_axes([0.13, 0.10, 0.55, 0.035])
        max_radius = max(5.0, 0.25 * min(
            scene.bounds[1] - scene.bounds[0], scene.bounds[3] - scene.bounds[2]
        ))
        self.radius_slider = Slider(
            slider_ax, "半径 (m)", 0.1, max_radius,
            valinit=np.clip(radius, 0.1, max_radius), valstep=0.05,
        )
        self.radius_slider.on_changed(self._radius_changed)

        self.save_button = Button(self.fig.add_axes([0.84, 0.82, 0.13, 0.055]), "保存 S")
        self.undo_button = Button(self.fig.add_axes([0.84, 0.74, 0.13, 0.055]), "撤销 Z")
        self.delete_button = Button(self.fig.add_axes([0.84, 0.66, 0.13, 0.055]), "删除 Del")
        self.clear_button = Button(self.fig.add_axes([0.84, 0.58, 0.13, 0.055]), "清空 C")
        self.save_button.on_clicked(lambda _event: self.save())
        self.undo_button.on_clicked(lambda _event: self.undo())
        self.delete_button.on_clicked(lambda _event: self.delete_selected())
        self.clear_button.on_clicked(lambda _event: self.clear())

        self.fig.canvas.mpl_connect("button_press_event", self._mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self._mouse_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._mouse_move)
        self.fig.canvas.mpl_connect("scroll_event", self._scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._key_press)
        self.redraw()

    def _snapshot(self) -> None:
        self.history.append((
            [item.copy() for item in self.scene.obstacles],
            self.scene.start_pos.copy(),
            self.scene.goal_pos.copy(),
        ))
        if len(self.history) > 100:
            self.history.pop(0)

    def _inside_bounds(self, x: float, y: float, radius: float = 0.0) -> bool:
        xmin, xmax, ymin, ymax = self.scene.bounds
        return (
            xmin + radius <= x <= xmax - radius
            and ymin + radius <= y <= ymax - radius
        )

    def _nearest_obstacle(self, x: float, y: float) -> Optional[int]:
        if not self.scene.obstacles:
            return None
        data = np.asarray(self.scene.obstacles, dtype=float)
        distances = np.linalg.norm(data[:, :2] - [x, y], axis=1)
        inside = np.flatnonzero(distances <= data[:, 2])
        if len(inside) == 0:
            return None
        return int(inside[np.argmin(distances[inside])])

    def _mouse_press(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        if event.button == 3:
            index = self._nearest_obstacle(x, y)
            if index is not None:
                self.selected = index
                self.delete_selected()
            return
        if event.button != 1:
            return

        if self.place_mode == "start":
            self._snapshot()
            self.scene.start_pos[:] = [x, y]
            self.place_mode = "obstacle"
            self.redraw()
            return
        if self.place_mode == "goal":
            self._snapshot()
            self.scene.goal_pos[:] = [x, y]
            self.place_mode = "obstacle"
            self.redraw()
            return

        index = self._nearest_obstacle(x, y)
        if index is None:
            radius = self.default_radius
            if not self._inside_bounds(x, y, radius):
                self._status("圆不能超出地图边界", error=True)
                return
            self._snapshot()
            self.scene.obstacles.append([x, y, radius])
            self.selected = len(self.scene.obstacles) - 1
            self.dragging = True
            self.click_offset[:] = 0.0
        else:
            self._snapshot()
            self.selected = index
            center = np.asarray(self.scene.obstacles[index][:2])
            self.click_offset = np.asarray([x, y]) - center
            self.dragging = True
            radius = self.scene.obstacles[index][2]
            self.radius_slider.set_val(radius)
        self.redraw()

    def _mouse_move(self, event) -> None:
        if not self.dragging or self.selected is None or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = np.asarray([event.xdata, event.ydata]) - self.click_offset
        radius = self.scene.obstacles[self.selected][2]
        xmin, xmax, ymin, ymax = self.scene.bounds
        x = float(np.clip(x, xmin + radius, xmax - radius))
        y = float(np.clip(y, ymin + radius, ymax - radius))
        self.scene.obstacles[self.selected][:2] = [x, y]
        self.redraw()

    def _mouse_release(self, _event) -> None:
        self.dragging = False

    def _radius_changed(self, value: float) -> None:
        self.default_radius = float(value)
        if self.selected is None:
            self._status(f"新障碍物半径：{value:.2f} m")
            return
        x, y, _old = self.scene.obstacles[self.selected]
        if not self._inside_bounds(x, y, value):
            return
        self.scene.obstacles[self.selected][2] = float(value)
        self.redraw()

    def _scroll(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        increment = 0.1 if event.button == "up" else -0.1
        self.radius_slider.set_val(np.clip(
            self.radius_slider.val + increment,
            self.radius_slider.valmin,
            self.radius_slider.valmax,
        ))

    def _key_press(self, event) -> None:
        key = (event.key or "").lower()
        if key == "s":
            self.save()
        elif key in {"delete", "backspace", "d"}:
            self.delete_selected()
        elif key == "z":
            self.undo()
        elif key == "c":
            self.clear()
        elif key == "1":
            self.place_mode = "start"
            self._status("请在地图内点击设置起点")
        elif key == "2":
            self.place_mode = "goal"
            self._status("请在地图内点击设置终点")
        elif key == "escape":
            self.selected = None
            self.place_mode = "obstacle"
            self.redraw()
        elif key in {"[", "-"}:
            self.radius_slider.set_val(max(
                self.radius_slider.valmin, self.radius_slider.val - 0.1
            ))
        elif key in {"]", "+", "="}:
            self.radius_slider.set_val(min(
                self.radius_slider.valmax, self.radius_slider.val + 0.1
            ))

    def delete_selected(self) -> None:
        if self.selected is None or not self.scene.obstacles:
            self._status("尚未选中圆形障碍物", error=True)
            return
        self._snapshot()
        del self.scene.obstacles[self.selected]
        self.selected = None
        self.redraw()

    def clear(self) -> None:
        if not self.scene.obstacles:
            return
        self._snapshot()
        self.scene.obstacles.clear()
        self.selected = None
        self.redraw()

    def undo(self) -> None:
        if not self.history:
            self._status("没有可撤销的操作", error=True)
            return
        obstacles, start, goal = self.history.pop()
        self.scene.obstacles = obstacles
        self.scene.start_pos = start
        self.scene.goal_pos = goal
        self.selected = None
        self.redraw()

    def save(self) -> None:
        save_scene(self.output, self.scene, self.resolution)
        self._status(f"已保存：{self.output}")

    def _status(self, message: str, error: bool = False) -> None:
        self.fig.suptitle(message, color="crimson" if error else "darkgreen", fontsize=11)
        self.fig.canvas.draw_idle()

    def redraw(self) -> None:
        self.ax.clear()
        xmin, xmax, ymin, ymax = self.scene.bounds
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.25)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title(
            f"左键添加/拖动，滚轮调半径，右键删除 | 障碍物：{len(self.scene.obstacles)}"
        )

        for index, (x, y, radius) in enumerate(self.scene.obstacles):
            selected = index == self.selected
            patch = Circle(
                (x, y), radius,
                facecolor="orange" if selected else "lightcoral",
                edgecolor="darkorange" if selected else "red",
                linewidth=2.5 if selected else 1.5,
                alpha=0.65,
            )
            self.ax.add_patch(patch)
            self.ax.text(x, y, f"{index}\nr={radius:.2f}", ha="center", va="center", fontsize=8)

        self.ax.plot(*self.scene.start_pos, marker="*", color="limegreen", markersize=15, label="起点 (键1)")
        self.ax.plot(*self.scene.goal_pos, marker="*", color="red", markersize=15, label="终点 (键2)")
        self.ax.legend(loc="upper left")
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        print("操作：左键添加/选中并拖动；滚轮或滑块调半径；右键删除")
        print("快捷键：S保存，Z撤销，C清空，1设置起点，2设置终点，Esc取消选择")
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="交互式圆形障碍物NPZ地图编辑器")
    parser.add_argument("--input", type=Path, help="打开已有NPZ继续编辑")
    parser.add_argument("--output", type=Path, default=Path("circle_scene.npz"))
    parser.add_argument(
        "--bounds", type=float, nargs=4, default=(0.0, 40.0, 0.0, 40.0),
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
    )
    parser.add_argument("--start", type=float, nargs=2, default=(5.0, 20.0))
    parser.add_argument("--goal", type=float, nargs=2, default=(35.0, 20.0))
    parser.add_argument("--radius", type=float, default=1.0, help="默认圆半径(m)")
    parser.add_argument("--resolution", type=float, default=0.05, help="地图分辨率(m/pixel)")
    parser.add_argument("--description", default="交互式圆形障碍物地图")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.radius <= 0 or args.resolution <= 0:
        raise SystemExit("--radius和--resolution必须大于0")

    if args.input:
        scene = load_scene(args.input)
        if args.output == Path("circle_scene.npz"):
            args.output = args.input
    else:
        bounds = np.asarray(args.bounds, dtype=float)
        if bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
            raise SystemExit("bounds必须满足xmax>xmin且ymax>ymin")
        scene = Scene(
            bounds=bounds,
            start_pos=np.asarray(args.start, dtype=float),
            goal_pos=np.asarray(args.goal, dtype=float),
            description=args.description,
        )

    CircleMapEditor(
        scene=scene,
        output=args.output,
        radius=args.radius,
        resolution=args.resolution,
    ).show()


if __name__ == "__main__":
    main()
