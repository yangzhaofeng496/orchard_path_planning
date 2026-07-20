"""Project-specific g2o adapter for :mod:`local_path_planning.teb.teb`."""

from types import SimpleNamespace

import numpy as np

from . import _g2o_teb_native


def _penalty(values, scale=100000000.0):
    values = np.asarray(values, dtype=float)
    return np.sqrt(scale) * np.maximum(0.0, -values)


def solve(planner, x0, bounds, obstacles):
    """Optimize the existing TEB objective through a native g2o LM graph."""
    lower = np.asarray([item[0] for item in bounds], dtype=float)
    upper = np.asarray([item[1] for item in bounds], dtype=float)

    def residual(variables):
        variables = np.asarray(variables, dtype=float)
        objective = max(0.0, float(planner._objective_function(variables, obstacles)))
        parts = [np.asarray([np.sqrt(objective)], dtype=float)]
        for constraint in (
            planner._speed_constraint,
            planner._acceleration_constraint,
            planner._steering_constraint,
            planner._forward_constraint,
            planner._progress_constraint,
        ):
            parts.append(_penalty(constraint(variables)))
        return np.concatenate(parts)

    residual_dimension = int(residual(np.asarray(x0, dtype=float)).size)
    raw = _g2o_teb_native.optimize(
        np.asarray(x0, dtype=float), lower, upper, residual,
        residual_dimension, int(planner.config.max_iterations),
    )
    return SimpleNamespace(**raw)
