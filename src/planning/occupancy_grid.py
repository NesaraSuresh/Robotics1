"""Log-odds occupancy grid built online from the 2D LiDAR.

The grid starts empty (unknown = free, optimistic). Each scan is projected into
the world using the *estimated* pose from the EKF and the fixed lidar extrinsic,
so mapping quality is itself a test of the estimator.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from src.config import PlannerConfig
from src.hal.interface import LaserScan

LIDAR_POS_B = np.array([-0.049, 0.0])   # lidar_site in base frame (x, y)


class OccupancyGrid:
    def __init__(self, cfg: PlannerConfig, bounds, margin: float = 0.25):
        self.cfg = cfg
        self.res = cfg.resolution
        self.x0, self.y0 = bounds[0] - margin, bounds[2] - margin
        self.nx = int(math.ceil((bounds[1] - bounds[0] + 2 * margin) / self.res))
        self.ny = int(math.ceil((bounds[3] - bounds[2] + 2 * margin) / self.res))
        self.logodds = np.zeros((self.ny, self.nx), dtype=np.float32)
        r = int(math.ceil(cfg.inflation / self.res))
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        self._inflated = None
        self._dist = None

    # ------------------------------------------------------------ indexing
    def to_cell(self, x, y):
        return (np.floor((np.asarray(y) - self.y0) / self.res).astype(int),
                np.floor((np.asarray(x) - self.x0) / self.res).astype(int))

    def to_world(self, row, col):
        return self.x0 + (np.asarray(col) + 0.5) * self.res, self.y0 + (np.asarray(row) + 0.5) * self.res

    def in_bounds(self, row, col):
        return (row >= 0) & (row < self.ny) & (col >= 0) & (col < self.nx)

    # -------------------------------------------------------------- update
    def update(self, scan: LaserScan, pose: np.ndarray, free_range_inf: float = 2.5) -> np.ndarray:
        """Integrate one scan; returns hit points in W (for display)."""
        x, y, th = pose
        c, s = math.cos(th), math.sin(th)
        ox = x + c * LIDAR_POS_B[0] - s * LIDAR_POS_B[1]
        oy = y + s * LIDAR_POS_B[0] + c * LIDAR_POS_B[1]
        ang = th + scan.angles
        rng = scan.ranges
        hit = np.isfinite(rng)
        r_free = np.where(hit, rng - self.res, free_range_inf)
        ds = np.arange(0.0, max(free_range_inf, float(np.nanmax(np.where(hit, rng, 0)))), self.res * 0.5)
        mask = ds[None, :] < r_free[:, None]
        fx = ox + np.cos(ang)[:, None] * ds[None, :]
        fy = oy + np.sin(ang)[:, None] * ds[None, :]
        fr, fc = self.to_cell(fx[mask], fy[mask])
        ok = self.in_bounds(fr, fc)
        free_idx = np.unique(fr[ok] * self.nx + fc[ok])
        hx = ox + np.cos(ang[hit]) * rng[hit]
        hy = oy + np.sin(ang[hit]) * rng[hit]
        hr, hc = self.to_cell(hx, hy)
        ok = self.in_bounds(hr, hc)
        hit_idx = np.unique(hr[ok] * self.nx + hc[ok])
        free_idx = np.setdiff1d(free_idx, hit_idx, assume_unique=True)
        flat = self.logodds.reshape(-1)
        flat[free_idx] += self.cfg.l_free
        flat[hit_idx] += self.cfg.l_occ
        np.clip(self.logodds, self.cfg.l_min, self.cfg.l_max, out=self.logodds)
        self._inflated = None
        return np.stack([hx, hy], axis=1)

    # ------------------------------------------------------------- queries
    def occupied(self) -> np.ndarray:
        return self.logodds > self.cfg.occ_threshold

    def inflated(self) -> np.ndarray:
        if self._inflated is None:
            occ = self.occupied().astype(np.uint8)
            occ[0, :] = occ[-1, :] = 1
            occ[:, 0] = occ[:, -1] = 1
            self._inflated = cv2.dilate(occ, self._kernel).astype(bool)
            free = (~self._inflated).astype(np.uint8)
            self._dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * self.res
        return self._inflated

    def clearance(self) -> np.ndarray:
        """Distance [m] from each free cell to the nearest inflated obstacle."""
        self.inflated()
        return self._dist

    def is_free_world(self, x, y) -> np.ndarray:
        r, c = self.to_cell(x, y)
        ok = self.in_bounds(r, c)
        out = np.zeros(np.shape(r), dtype=bool)
        inf = self.inflated()
        out[ok] = ~inf[r[ok], c[ok]]
        return out
