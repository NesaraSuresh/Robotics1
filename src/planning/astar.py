"""A* on the inflated occupancy grid (8-connected, octile heuristic).

Acknowledgement: the graph-search planner structure (grid nodes on a map,
A* with a distance heuristic, start/goal snapped to the nearest free node) was
inspired by the path_planning.py module of the Smart_Factory project
(https://github.com/Prethivi1608/Smart_Factory, ROS 2). This is an independent
re-implementation for our MuJoCo pipeline: heap-based open set, clearance cost
term, and no ROS dependency.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

import numpy as np

SQRT2 = math.sqrt(2.0)
NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def nearest_free(blocked: np.ndarray, cell: Tuple[int, int], max_r: int = 20) -> Optional[Tuple[int, int]]:
    r0, c0 = cell
    ny, nx = blocked.shape
    for rad in range(0, max_r + 1):
        best = None
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if max(abs(dr), abs(dc)) != rad:
                    continue
                r, c = r0 + dr, c0 + dc
                if 0 <= r < ny and 0 <= c < nx and not blocked[r, c]:
                    d = dr * dr + dc * dc
                    if best is None or d < best[0]:
                        best = (d, (r, c))
        if best:
            return best[1]
    return None


def astar(blocked: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
          clearance: Optional[np.ndarray] = None, clearance_weight: float = 2.0,
          clearance_pref: float = 0.4, res: float = 0.05) -> Optional[List[Tuple[int, int]]]:
    """Returns the list of (row, col) cells from start to goal, or None."""
    ny, nx = blocked.shape
    start = nearest_free(blocked, start)
    goal = nearest_free(blocked, goal)
    if start is None or goal is None:
        return None
    pref_cells = clearance_pref / res

    def h(rc):
        dr, dc = abs(rc[0] - goal[0]), abs(rc[1] - goal[1])
        return (dr + dc) + (SQRT2 - 2) * min(dr, dc)

    g = np.full((ny, nx), np.inf)
    parent = -np.ones((ny, nx), dtype=np.int64)
    closed = np.zeros((ny, nx), dtype=bool)
    g[start] = 0.0
    heap = [(h(start), 0.0, start)]
    while heap:
        _, gc, cur = heapq.heappop(heap)
        if closed[cur]:
            continue
        closed[cur] = True
        if cur == goal:
            break
        r, c = cur
        for dr, dc, step in NEIGHBOURS:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < ny and 0 <= cc < nx) or blocked[rr, cc] or closed[rr, cc]:
                continue
            cost = step
            if clearance is not None:
                d = clearance[rr, cc] / res
                if d < pref_cells:
                    cost += clearance_weight * (pref_cells - d) / pref_cells
            ng = gc + cost
            if ng < g[rr, cc]:
                g[rr, cc] = ng
                parent[rr, cc] = r * nx + c
                heapq.heappush(heap, (ng + h((rr, cc)), ng, (rr, cc)))
    if not closed[goal]:
        return None
    path = [goal]
    while path[-1] != start:
        p = parent[path[-1]]
        path.append((int(p // nx), int(p % nx)))
    return path[::-1]
