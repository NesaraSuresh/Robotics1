"""Grid path -> smooth, time-parameterised reference for the LQR tracker.

Steps: line-of-sight shortcutting -> resample every 5 cm -> iterative smoothing
(data/smoothness weighted, collision-checked) -> heading, curvature, arc length,
speed profile v(s) = min(v_nav, sqrt(a_lat/|kappa|), sqrt(2 a_long (s_end - s))).

The reference at the robot's projection onto the path gives (x_r, y_r, th_r, v_r, w_r)
with w_r = v_r * kappa -- the feed-forward the LQR linearises about.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.estimation.ekf import wrap
from src.planning.occupancy_grid import OccupancyGrid


def _line_free(grid: OccupancyGrid, p, q) -> bool:
    n = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / (grid.res * 0.5)))
    t = np.linspace(0, 1, n)
    return bool(np.all(grid.is_free_world(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))))


def shortcut(grid: OccupancyGrid, pts: np.ndarray) -> np.ndarray:
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _line_free(grid, pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return np.array(out)


def resample(pts: np.ndarray, ds: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return pts[:1].repeat(2, axis=0)
    n = max(2, int(math.ceil(s[-1] / ds)) + 1)
    si = np.linspace(0, s[-1], n)
    return np.stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])], axis=1)


def smooth(grid: Optional[OccupancyGrid], pts: np.ndarray, iters: int = 60,
           w_data: float = 0.15, w_smooth: float = 0.35) -> np.ndarray:
    p = pts.copy()
    for _ in range(iters):
        new = p.copy()
        new[1:-1] += w_data * (pts[1:-1] - p[1:-1]) + w_smooth * (p[:-2] + p[2:] - 2 * p[1:-1])
        if grid is not None:
            ok = grid.is_free_world(new[:, 0], new[:, 1])
            new[~ok] = p[~ok]
        p = new
    return p


@dataclass
class ReferencePoint:
    x: float
    y: float
    theta: float
    v: float
    w: float
    kappa: float
    s: float
    remaining: float
    index: int


class ReferencePath:
    def __init__(self, pts: np.ndarray, v_nav: float, a_lat: float, a_long: float,
                 v_end: float = 0.0, v_min: float = 0.03, final_heading: Optional[float] = None):
        self.pts = pts
        d = np.diff(pts, axis=0)
        seg = np.linalg.norm(d, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        th = np.arctan2(d[:, 1], d[:, 0])
        th = np.concatenate([th, [th[-1] if final_heading is None else final_heading]])
        self.theta = np.unwrap(th)
        # curvature d(theta)/ds, lightly smoothed
        k = np.gradient(self.theta, self.s, edge_order=1) if len(pts) > 2 else np.zeros(len(pts))
        if len(k) >= 5:
            k = np.convolve(k, np.ones(5) / 5, mode="same")
        self.kappa = k
        self.length = float(self.s[-1])
        v_curv = np.sqrt(a_lat / np.maximum(np.abs(k), 1e-3))
        v_stop = np.sqrt(v_end ** 2 + 2 * a_long * np.maximum(self.length - self.s, 0.0))
        self.v = np.clip(np.minimum.reduce([np.full_like(k, v_nav), v_curv, v_stop]), v_min, v_nav)
        self.v_min = v_min
        self._idx = 0

    def project(self, p: np.ndarray, window: int = 40) -> int:
        lo = max(0, self._idx - 5)
        hi = min(len(self.pts), self._idx + window)
        d = np.linalg.norm(self.pts[lo:hi] - p[:2], axis=1)
        self._idx = lo + int(np.argmin(d))
        return self._idx

    def reference(self, pose: np.ndarray) -> ReferencePoint:
        """Continuous projection onto the polyline (interpolated between vertices), so the
        along-track error does not jump by the 5 cm vertex spacing."""
        i = self.project(pose)
        best = (float("inf"), i, 0.0)
        for a in (i - 1, i):
            if a < 0 or a + 1 >= len(self.pts):
                continue
            seg = self.pts[a + 1] - self.pts[a]
            L2 = float(seg @ seg)
            t = 0.0 if L2 < 1e-12 else float(np.clip((pose[:2] - self.pts[a]) @ seg / L2, 0.0, 1.0))
            d = float(np.linalg.norm(self.pts[a] + t * seg - pose[:2]))
            if d < best[0]:
                best = (d, a, t)
        _, a, t = best
        b = min(a + 1, len(self.pts) - 1)
        lerp = lambda arr: float((1 - t) * arr[a] + t * arr[b])  # noqa: E731
        v = lerp(self.v)
        k = lerp(self.kappa)
        s_ref = lerp(self.s)
        return ReferencePoint(lerp(self.pts[:, 0]), lerp(self.pts[:, 1]), float(wrap(lerp(self.theta))),
                              v, v * k, k, s_ref, self.length - s_ref, a)

    def cross_track(self, p: np.ndarray) -> float:
        """Unsigned distance from p to the polyline (evaluation)."""
        a, b = self.pts[:-1], self.pts[1:]
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", p[:2] - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
        proj = a + t[:, None] * ab
        return float(np.min(np.linalg.norm(proj - p[:2], axis=1)))

    def blocked(self, grid: OccupancyGrid, from_index: int = 0) -> bool:
        p = self.pts[from_index:]
        return bool(np.any(~grid.is_free_world(p[:, 0], p[:, 1])))


def plan_path(grid: OccupancyGrid, start_xy, goal_xy, cfg_planner, cfg_mission,
              final_heading: Optional[float] = None) -> Optional[ReferencePath]:
    from src.planning.astar import astar
    inflated = grid.inflated()
    s = grid.to_cell(start_xy[0], start_xy[1])
    g = grid.to_cell(goal_xy[0], goal_xy[1])
    cells = astar(inflated, (int(s[0]), int(s[1])), (int(g[0]), int(g[1])), grid.clearance(),
                  res=grid.res)
    if cells is None:
        return None
    rows, cols = np.array(cells).T
    wx, wy = grid.to_world(rows, cols)
    pts = np.stack([wx, wy], axis=1)
    pts[0] = start_xy[:2]
    pts[-1] = goal_xy[:2]
    pts = shortcut(grid, pts)
    pts = resample(pts, 0.05)
    pts = smooth(grid, pts, cfg_planner.smooth_iterations)
    pts = resample(pts, 0.05)
    if len(pts) < 3:
        pts = resample(np.array([start_xy[:2], goal_xy[:2]]), 0.05)
    return ReferencePath(pts, cfg_mission.v_nav, cfg_mission.a_lat_max, cfg_mission.a_long_max,
                         v_end=0.05, final_heading=final_heading)
