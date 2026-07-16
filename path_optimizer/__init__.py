"""
Path Optimizer Module

提供路径后处理优化功能，包括：
- Path Shortcut（路径捷径优化）
- 冗余节点删除
- 共线点移除
"""

from .shortcut import ShortcutOptimizer, CollisionChecker
from .curvature_smoother import CurvatureSmoother, SmoothedPath

__all__ = [
    "ShortcutOptimizer",
    "CollisionChecker",
    "CurvatureSmoother",
    "SmoothedPath",
]
