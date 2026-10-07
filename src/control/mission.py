"""Docking mission state machine. Every decision uses the EKF estimate only.

    NAVIGATE    A* path on the LiDAR occupancy grid to the pre-dock pose, LQR tracking,
                replan when the path becomes blocked / the robot deviates / the path ends.
    ALIGN_PATH  rotate in place to the path heading when the initial heading error is large.
    ALIGN_DOCK  at the pre-dock pose, rotate to face the dock marker.
    DOCK        straight-line reference along the dock axis, speed proportional to the
                remaining distance, LQR tracking; stop at the dock pose.
    DOCKED / FAILED  terminal.
LiDAR safety layer: forward motion is blocked if a return is inside the front sector
closer than `safety_distance` from the robot's front (rotation still allowed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from src.config import Config
from src.control.lqr_tracker import LQRTracker
from src.estimation.ekf import wrap
from src.hal.interface import LaserScan
from src.planning.occupancy_grid import LIDAR_POS_B, OccupancyGrid
from src.planning.path import ReferencePath, plan_path, resample


class Mode(str, Enum):
    NAVIGATE = "NAVIGATE"
    ALIGN_PATH = "ALIGN_PATH"
    ALIGN_DOCK = "ALIGN_DOCK"
    DOCK = "DOCK"
    DOCKED = "DOCKED"
    FAILED = "FAILED"


@dataclass
class Command:
    v: float
    w: float
    mode: Mode
    e: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_ref: float = 0.0
    w_ref: float = 0.0
    safety_stop: bool = False
    replanned: bool = False


class DockingMission:
    def __init__(self, cfg: Config, grid: OccupancyGrid, tracker: LQRTracker,
                 predock: np.ndarray, dock: np.ndarray):
        self.cfg = cfg
        self.grid = grid
        self.tracker = tracker
        self.predock = predock
        self.dock = dock
        self.mode = Mode.NAVIGATE
        self.path: Optional[ReferencePath] = None
        self.dock_path: Optional[ReferencePath] = None
        self._last_plan_t = -1e9
        self._last_check_t = -1e9
        self.n_replans = 0
        self.n_safety_stops = 0
        self._plan_failures = 0

    # ------------------------------------------------------------ helpers
    def _front_blocked(self, scan: Optional[LaserScan]) -> bool:
        if scan is None:
            return getattr(self, "_blocked_latch", False)
        m = self.cfg.mission
        half = math.radians(m.safety_half_angle_deg)
        sector = np.abs(scan.angles) < half
        # distance from robot front to the hit, along the lidar ray (lidar is behind the front)
        front_offset = self.cfg.robot.front_offset - LIDAR_POS_B[0]
        d = scan.ranges[sector] * np.cos(scan.angles[sector]) - front_offset
        self._blocked_latch = bool(np.any(d < m.safety_distance))
        return self._blocked_latch

    def _replan(self, t: float, pose: np.ndarray) -> bool:
        self._last_plan_t = t
        path = plan_path(self.grid, pose[:2], self.predock[:2], self.cfg.planner, self.cfg.mission,
                         final_heading=None)
        if path is None:
            self._plan_failures += 1
            return False
        self.path = path
        self.n_replans += 1
        self._plan_failures = 0
        ref = path.reference(pose)
        if abs(wrap(ref.theta - pose[2])) > 0.8:
            self.mode = Mode.ALIGN_PATH
        return True

    def _make_dock_path(self, pose: np.ndarray) -> ReferencePath:
        yaw = self.dock[2]
        d = np.array([math.cos(yaw), math.sin(yaw)])
        # start the line at the robot's projection onto the dock axis (behind it slightly)
        s0 = float(np.dot(pose[:2] - self.dock[:2], d)) - 0.05
        start = self.dock[:2] + s0 * d
        pts = resample(np.array([start, self.dock[:2]]), 0.02)
        return ReferencePath(pts, self.cfg.mission.v_dock_max, 1.0, 1.0, v_end=0.0,
                             v_min=self.cfg.mission.v_dock_min, final_heading=yaw)

    def dock_remaining(self, pose: np.ndarray) -> float:
        yaw = self.dock[2]
        return float(np.dot(self.dock[:2] - pose[:2], [math.cos(yaw), math.sin(yaw)]))

    # --------------------------------------------------------------- step
    def step(self, t: float, pose: np.ndarray, scan: Optional[LaserScan]) -> Command:
        m, p = self.cfg.mission, self.cfg.planner

        if self.mode in (Mode.NAVIGATE, Mode.ALIGN_PATH):
            if np.linalg.norm(pose[:2] - self.predock[:2]) < m.predock_tolerance:
                self.mode = Mode.ALIGN_DOCK
            else:
                replanned = False
                need = self.path is None
                if not need and t - self._last_check_t > 0.5:
                    self._last_check_t = t
                    i = self.path.project(pose)
                    dev = np.linalg.norm(self.path.pts[i] - pose[:2])
                    need = (self.path.blocked(self.grid, i) or dev > p.max_deviation
                            or (self.path.length - self.path.s[i] < 0.05))
                if need and t - self._last_plan_t > (0.3 if self.path is None else p.replan_period):
                    replanned = self._replan(t, pose)
                if self.path is None:
                    out = self.tracker.rotate_to(pose, pose[2] + 0.6)   # look around, retry
                    return Command(out.v, out.w, self.mode, out.e, replanned=replanned)
                ref = self.path.reference(pose)
                if self.mode == Mode.ALIGN_PATH:
                    out = self.tracker.rotate_to(pose, ref.theta)
                    if abs(out.e[2]) < 0.15:
                        self.mode = Mode.NAVIGATE
                    return Command(out.v, out.w, Mode.ALIGN_PATH, out.e, replanned=replanned)
                out = self.tracker.control(pose, ref)
                blocked = self._front_blocked(scan)
                v = min(out.v, 0.0) if blocked else out.v
                if blocked:
                    self.n_safety_stops += 1
                    self._last_check_t = -1e9      # force a blocked-path check next step
                return Command(v, out.w, Mode.NAVIGATE, out.e, ref.v, ref.w, blocked, replanned)

        if self.mode == Mode.ALIGN_DOCK:
            out = self.tracker.rotate_to(pose, self.dock[2])
            if abs(out.e[2]) < m.align_tolerance:
                self.mode = Mode.DOCK
                self.dock_path = self._make_dock_path(pose)
            return Command(out.v, out.w, Mode.ALIGN_DOCK, out.e)

        if self.mode == Mode.DOCK:
            rem = self.dock_remaining(pose)
            if rem <= 0.005:
                self.mode = Mode.DOCKED
                return Command(0.0, 0.0, Mode.DOCKED)
            ref = self.dock_path.reference(pose)
            ref.v = float(np.clip(m.k_dock_speed * rem, m.v_dock_min, m.v_dock_max))
            ref.w = 0.0
            out = self.tracker.control(pose, ref)
            return Command(max(out.v, 0.0), out.w, Mode.DOCK, out.e, ref.v, ref.w)

        return Command(0.0, 0.0, self.mode)
