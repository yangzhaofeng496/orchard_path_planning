#!/usr/bin/env python3
"""Informed RRT*: RRT* with prolate-hyperspheroid sampling after first solution."""
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


def informed_sample(rng, start, goal, best_cost, bounds):
    direct = math.dist(start, goal)
    if not math.isfinite(best_cost) or best_cost <= direct + 1e-12:
        xmin, xmax, ymin, ymax = bounds
        return float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax))
    centre = ((start[0] + goal[0]) / 2.0, (start[1] + goal[1]) / 2.0)
    major, minor = best_cost / 2.0, math.sqrt(max(0.0, best_cost * best_cost - direct * direct)) / 2.0
    angle = math.atan2(goal[1] - start[1], goal[0] - start[0])
    radius, theta = math.sqrt(float(rng.random())), float(rng.uniform(0.0, 2.0 * math.pi))
    ux, uy = radius * math.cos(theta), radius * math.sin(theta)
    x = centre[0] + major * ux * math.cos(angle) - minor * uy * math.sin(angle)
    y = centre[1] + major * ux * math.sin(angle) + minor * uy * math.cos(angle)
    xmin, xmax, ymin, ymax = bounds
    return min(xmax, max(xmin, x)), min(ymax, max(ymin, y))


def extract(nodes, parent, goal):
    path = [goal]
    while parent is not None:
        path.append(nodes[parent].point); parent = nodes[parent].parent
    return list(reversed(path))


def plan(start, goal, bounds, checker, obstacles, seed, args):
    del obstacles
    rng = np.random.default_rng(seed)
    nodes = [Node(start, None, 0.0)]
    best_parent, best_cost, first = None, math.inf, -1
    for iteration in range(1, args.max_iterations + 1):
        sample = informed_sample(rng, start, goal, best_cost, bounds)
        nearest = min(range(len(nodes)), key=lambda i: math.dist(nodes[i].point, sample))
        source, distance = nodes[nearest].point, math.dist(nodes[nearest].point, sample)
        if distance <= 1e-12:
            continue
        scale = min(1.0, args.step_size / distance)
        new = (source[0] + (sample[0] - source[0]) * scale,
               source[1] + (sample[1] - source[1]) * scale)
        if math.dist(start, new) + math.dist(new, goal) >= best_cost or not checker.check_line(source, new):
            continue
        radius = min(args.near_radius,
                     args.rewire_gamma * math.sqrt(math.log(len(nodes) + 1) / (len(nodes) + 1)))
        near = [i for i, node in enumerate(nodes) if math.dist(node.point, new) <= radius]
        parent, cost = nearest, nodes[nearest].cost + math.dist(source, new)
        for i in near:
            candidate = nodes[i].cost + math.dist(nodes[i].point, new)
            if candidate < cost and candidate + math.dist(new, goal) < best_cost and checker.check_line(nodes[i].point, new):
                parent, cost = i, candidate
        nodes.append(Node(new, parent, cost)); new_i = len(nodes) - 1
        for i in near:
            candidate = cost + math.dist(new, nodes[i].point)
            if candidate + 1e-12 < nodes[i].cost and checker.check_line(new, nodes[i].point):
                nodes[i].parent, nodes[i].cost = new_i, candidate
        if math.dist(new, goal) <= args.goal_connect_distance and checker.check_line(new, goal):
            candidate = cost + math.dist(new, goal)
            if candidate < best_cost:
                best_parent, best_cost = new_i, candidate
                if first < 0:
                    first = iteration
    return SearchResult(extract(nodes, best_parent, goal) if best_parent is not None else None,
                        len(nodes), first)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--step-size", type=float, default=3.0)
    parser.add_argument("--near-radius", type=float, default=7.0)
    parser.add_argument("--rewire-gamma", type=float, default=45.0)
    parser.add_argument("--goal-connect-distance", type=float, default=5.0)
    args = parser.parse_args()
    run_benchmark(args, "Informed RRT*", plan)


if __name__ == "__main__":
    main()
