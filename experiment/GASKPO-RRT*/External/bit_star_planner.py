#!/usr/bin/env python3
"""Batch Informed Trees (BIT*) with lazy edge ordering and informed batches."""
import argparse
import heapq
import math
from dataclasses import dataclass

import numpy as np

from _planner_common import SearchResult, add_common_arguments, run_benchmark
from informed_rrt_star_planner import informed_sample


@dataclass
class Vertex:
    point: tuple[float, float]
    parent: int | None
    cost: float


def plan(start, goal, bounds, checker, obstacles, seed, args):
    del obstacles
    rng = np.random.default_rng(seed)
    vertices = [Vertex(start, None, 0.0)]
    samples = []
    best_parent, best_cost, first = None, math.inf, -1
    edge_evaluations = 0
    batch = 0
    while edge_evaluations < args.max_iterations:
        batch += 1
        # Prune samples that cannot improve the incumbent and add a new batch.
        samples = [p for p in samples if math.dist(start, p) + math.dist(p, goal) < best_cost]
        for _ in range(args.batch_size):
            for _attempt in range(100):
                point = informed_sample(rng, start, goal, best_cost, bounds)
                if checker.point_free(point) and math.dist(start, point) + math.dist(point, goal) < best_cost:
                    samples.append(point); break
        samples.append(goal)
        radius = min(args.near_radius,
                     args.rewire_gamma * math.sqrt(max(math.log(len(vertices) + len(samples) + 1), 1.0) /
                                                   (len(vertices) + len(samples) + 1)))
        vertex_queue = [(v.cost + math.dist(v.point, goal), i) for i, v in enumerate(vertices)
                        if v.cost + math.dist(v.point, goal) < best_cost]
        heapq.heapify(vertex_queue)
        edge_queue = []
        expanded = set()
        while (vertex_queue or edge_queue) and edge_evaluations < args.max_iterations:
            best_vertex_key = vertex_queue[0][0] if vertex_queue else math.inf
            best_edge_key = edge_queue[0][0] if edge_queue else math.inf
            if best_vertex_key <= best_edge_key:
                _, vi = heapq.heappop(vertex_queue)
                if vi in expanded:
                    continue
                expanded.add(vi); vertex = vertices[vi]
                # Lazily enqueue promising sample edges.
                for point in samples:
                    distance = math.dist(vertex.point, point)
                    estimate = vertex.cost + distance + math.dist(point, goal)
                    if distance <= radius and estimate < best_cost:
                        heapq.heappush(edge_queue, (estimate, vertex.cost + distance, vi, 0, -1, point))
                # Rewiring candidates are also ordered lazily.
                for wi, other in enumerate(vertices):
                    if wi == vi:
                        continue
                    distance = math.dist(vertex.point, other.point)
                    estimate = vertex.cost + distance + math.dist(other.point, goal)
                    if distance <= radius and vertex.cost + distance < other.cost and estimate < best_cost:
                        heapq.heappush(edge_queue, (estimate, vertex.cost + distance, vi, 1, wi, other.point))
                continue
            _, candidate_cost, vi, kind, target_i, target = heapq.heappop(edge_queue)
            source = vertices[vi]
            if kind:
                target = vertices[target_i].point
            elif target not in samples:
                continue
            if candidate_cost + math.dist(target, goal) >= best_cost:
                continue
            edge_evaluations += 1
            if not checker.check_line(source.point, target):
                continue
            if kind:
                if candidate_cost + 1e-12 < vertices[target_i].cost:
                    vertices[target_i].parent, vertices[target_i].cost = vi, candidate_cost
                    heapq.heappush(vertex_queue, (candidate_cost + math.dist(target, goal), target_i))
                continue
            if target == goal:
                if candidate_cost < best_cost:
                    best_parent, best_cost = vi, candidate_cost
                    if first < 0:
                        first = edge_evaluations
                continue
            # The sample may already have been consumed by another queued edge.
            vertices.append(Vertex(target, vi, candidate_cost)); new_i = len(vertices) - 1
            samples.remove(target)
            heapq.heappush(vertex_queue, (candidate_cost + math.dist(target, goal), new_i))
        if batch >= args.max_batches:
            break
    if best_parent is None:
        return SearchResult(None, len(vertices), -1)
    path, index = [goal], best_parent
    while index is not None:
        path.append(vertices[index].point); index = vertices[index].parent
    return SearchResult(list(reversed(path)), len(vertices), first)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-iterations", type=int, default=3000, help="Maximum collision-checked edges")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--near-radius", type=float, default=10.0)
    parser.add_argument("--rewire-gamma", type=float, default=70.0)
    args = parser.parse_args()
    run_benchmark(args, "BIT*", plan)


if __name__ == "__main__":
    main()
