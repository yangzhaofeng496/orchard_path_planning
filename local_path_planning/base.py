"""
局部路径规划器基类
定义统一接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class Pose:
    """位姿"""
    x: float
    y: float
    yaw: float


@dataclass
class VehicleState:
    """车辆状态"""
    x: float
    y: float
    yaw: float
    speed: float
    steering: float


@dataclass
class Control:
    """控制指令"""
    speed: float
    steering: float


@dataclass
class CircleObstacle:
    """圆形障碍物"""
    x: float
    y: float
    radius: float


@dataclass
class LocalPlanResult:
    """局部规划结果"""
    success: bool
    control: Optional[Control]
    trajectory: List[Pose]
    cost: float = float('inf')


class LocalPlannerConfig(ABC):
    """局部规划器配置基类"""
    pass


class LocalPlanner(ABC):
    """局部路径规划器基类"""

    def __init__(self, config: LocalPlannerConfig):
        self.config = config
        self.global_path: List[Pose] = []

    def set_global_path(self, path: List[Pose]):
        """设置全局路径"""
        self.global_path = path

    @abstractmethod
    def plan(
        self,
        state: VehicleState,
        obstacles: List[CircleObstacle],
    ) -> LocalPlanResult:
        """
        局部规划

        Args:
            state: 当前车辆状态
            obstacles: 障碍物列表

        Returns:
            规划结果
        """
        pass
