#!/usr/bin/env python3
"""Standard uniformly sampled RRT* batch planner for orchard NPZ maps."""
import argparse
import math
from dataclasses import dataclass

import numpy as np

from _planner_common import SearchResult, add_common_arguments, run_benchmark


@dataclass
class Node:
    point: tuple[float, float]
    parent: int | None
    cost: float


def plan(start, goal, bounds, checker, obstacles, seed, args):
    del obstacles
    rng = np.random.default_rng(seed)
    nodes = [Node(start, None, 0.0)]
    first = -1
    best_parent, best_cost = None, math.inf
    xmin, xmax, ymin, ymax = bounds
    for iteration in range(1, args.max_iterations + 1):
        sample = (float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax)))
        nearest = min(range(len(nodes)), key=lambda i: math.dist(nodes[i].point, sample))
        source = nodes[nearest].point
        distance = math.dist(source, sample)
        if distance <= 1e-12:
            continue
        ratio = min(1.0, args.step_size / distance)
        new_point = (source[0] + (sample[0] - source[0]) * ratio,
                     source[1] + (sample[1] - source[1]) * ratio)
        if not checker.check_line(source, new_point):
            continue
        radius = min(args.near_radius, args.rewire_gamma * math.sqrt(math.log(len(nodes) + 1) / (len(nodes) + 1)))
        near = [i for i, n in enumerate(nodes) if math.dist(n.point, new_point) <= radius]
        parent, cost = nearest, nodes[nearest].cost + math.dist(source, new_point)
        for i in near:
            candidate = nodes[i].cost + math.dist(nodes[i].point, new_point)
            if candidate < cost and checker.check_line(nodes[i].point, new_point):
                parent, cost = i, candidate
        nodes.append(Node(new_point, parent, cost)); new_i = len(nodes) - 1
        for i in near:
            candidate = cost + math.dist(new_point, nodes[i].point)
            if candidate + 1e-12 < nodes[i].cost and checker.check_line(new_point, nodes[i].point):
                nodes[i].parent, nodes[i].cost = new_i, candidate
        if math.dist(new_point, goal) <= args.goal_connect_distance and checker.check_line(new_point, goal):
            candidate = cost + math.dist(new_point, goal)
            if candidate < best_cost:
                best_parent, best_cost = new_i, candidate
                if first < 0: first = iteration
    if best_parent is None:
        return SearchResult(None, len(nodes), -1)
    path = [goal]
    index = best_parent
    while index is not None:
        path.append(nodes[index].point); index = nodes[index].parent
    return SearchResult(list(reversed(path)), len(nodes), first)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--step-size", type=float, default=3.0)
    parser.add_argument("--near-radius", type=float, default=7.0)
    parser.add_argument("--rewire-gamma", type=float, default=45.0)
    parser.add_argument("--goal-connect-distance", type=float, default=5.0)
    args = parser.parse_args()
    run_benchmark(args, "RRT*", plan)


if __name__ == "__main__":
    main()
