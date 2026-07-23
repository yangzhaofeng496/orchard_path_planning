#!/usr/bin/env python3
"""GASKPO-RRT* using the same GoalBias+Tangent setup as the ablation study."""
import argparse

from _planner_common import SearchResult, add_common_arguments, run_benchmark
from global_path_planning.innovation_sample.ackermann_rrt_star import AckermannRRTStar
from vehicle.reeds_shepp_path import Pose


def plan(start, goal, bounds, checker, obstacles, seed, args):
    """Run the exact GoalBias+Tangent planner configuration used by Ablation."""
    # batch_benchmark has separate ours_* options so aligning our method does
    # not silently change the competing RRT* configuration.  The standalone
    # entry point uses the unprefixed fallbacks below.
    max_iterations = getattr(args, "ours_max_iterations", args.max_iterations)
    expand_length = getattr(args, "ours_expand_length", args.step_size)
    integration_step = getattr(args, "ours_integration_step_size", args.integration_step_size)
    near_radius = getattr(args, "ours_near_radius", args.near_radius)
    goal_connect_distance = getattr(args, "ours_goal_connect_distance", args.goal_connect_distance)
    planner = AckermannRRTStar(
        start=Pose(float(start[0]), float(start[1]), 0.0),
        goal=Pose(float(goal[0]), float(goal[1]), 0.0),
        bounds=bounds,
        vehicle=checker.vehicle,
        obstacles=obstacles,
        curvature=1.0 / 3.0,
        use_ackermann_constraints=False,
        expand_length=expand_length,
        step_size=integration_step,
        goal_connect_distance=goal_connect_distance,
        max_iterations=max_iterations,
        near_radius=near_radius,
        use_hybrid_sampling=True,
        goal_probability=args.goal_bias,
        tangent_probability=args.tangent_probability,
        adaptive_sampling_probabilities=False,
        corridor_probability=0.0,
        rectangle_probability=0.45,
        allow_reverse=True,
        use_tangent_guidance=True,
        tangent_clearance=args.tangent_clearance,
        shrink_probability=0.35,
        shrink_length_factor=0.70,
        shrink_width_factor=0.70,
        shrink_activation_distance=18.0,
        near_anchor_probability=0.55,
        near_anchor_length_ratio=0.40,
        cluster_shape="ellipse",
        rectangle_anchor_mode="closest_to_goal",
        use_goal_connector=True,
        relax_goal_yaw=False,
        random_seed=seed,
    )
    result = planner.planning()
    if result is None:
        return SearchResult(None, len(planner.nodes), -1, planner.planning_time)
    path_x, path_y, _, _ = result
    path = list(zip(map(float, path_x), map(float, path_y)))
    first_iteration = planner.first_solution_iteration
    return SearchResult(path, len(planner.nodes),
                        int(first_iteration) if first_iteration is not None else -1,
                        planner.planning_time)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-iterations", type=int, default=2500)
    parser.add_argument("--step-size", type=float, default=3.0,
                        help="Tree expansion length (Ablation: 3.0 m)")
    parser.add_argument("--integration-step-size", type=float, default=0.08)
    parser.add_argument("--near-radius", type=float, default=5.0)
    parser.add_argument("--goal-connect-distance", type=float, default=7.0)
    parser.add_argument("--goal-bias", type=float, default=0.20)
    parser.add_argument("--tangent-probability", type=float, default=0.10)
    parser.add_argument("--tangent-clearance", type=float, default=0.50)
    args = parser.parse_args()
    if args.goal_bias < 0 or args.tangent_probability < 0 or args.goal_bias + args.tangent_probability > 1:
        parser.error("goal-bias and tangent-probability must be non-negative and sum to <= 1")
    run_benchmark(args, "GASKPO-RRT* (ours)", plan)


if __name__ == "__main__":
    main()
