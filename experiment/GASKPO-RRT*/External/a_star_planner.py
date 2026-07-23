#!/usr/bin/env python3
"""Batch A* planner for the orchard NPZ format."""
import argparse
import heapq
import math

from _planner_common import SearchResult, add_common_arguments, run_benchmark


def plan(start, goal, bounds, checker, obstacles, seed, args):
    del obstacles, seed
    resolution = args.grid_resolution
    xmin, xmax, ymin, ymax = bounds
    nx = int(math.floor((xmax - xmin) / resolution)) + 1
    ny = int(math.floor((ymax - ymin) / resolution)) + 1
    to_cell = lambda p: (min(nx - 1, max(0, round((p[0] - xmin) / resolution))),
                         min(ny - 1, max(0, round((p[1] - ymin) / resolution))))
    to_point = lambda c: (xmin + c[0] * resolution, ymin + c[1] * resolution)
    target = to_cell(goal)
    candidates = [(to_cell(start)[0] + dx, to_cell(start)[1] + dy)
                  for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    candidates = [c for c in candidates if 0 <= c[0] < nx and 0 <= c[1] < ny
                  and checker.check_line(start, to_point(c))]
    if not candidates:
        return SearchResult(None, 0, -1)
    source = min(candidates, key=lambda c: math.dist(start, to_point(c)))
    open_heap = [(math.dist(start, goal), 0.0, source)]
    cost, parent, closed = {source: 0.0}, {}, set()
    directions = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    expanded = 0
    while open_heap:
        _, g, cell = heapq.heappop(open_heap)
        if cell in closed or g > cost.get(cell, math.inf) + 1e-12:
            continue
        closed.add(cell); expanded += 1
        point = to_point(cell)
        if math.dist(point, goal) <= math.sqrt(2.0) * resolution and checker.check_line(point, goal):
            cells = [cell]
            while cells[-1] != source:
                cells.append(parent[cells[-1]])
            path = [start] + [to_point(c) for c in reversed(cells)] + [goal]
            return SearchResult(path, len(cost), expanded)
        for dx, dy in directions:
            nxt = (cell[0] + dx, cell[1] + dy)
            if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny) or nxt in closed:
                continue
            next_point = to_point(nxt)
            if not checker.check_line(point, next_point):
                continue
            new_cost = g + math.hypot(dx, dy) * resolution
            if new_cost + 1e-12 < cost.get(nxt, math.inf):
                cost[nxt], parent[nxt] = new_cost, cell
                heapq.heappush(open_heap, (new_cost + math.dist(next_point, goal), new_cost, nxt))
    return SearchResult(None, len(cost), -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--grid-resolution", type=float, default=1.0)
    args = parser.parse_args()
    run_benchmark(args, "A*", plan)


if __name__ == "__main__":
    main()
