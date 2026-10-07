"""Ground-truth oracle -- EVALUATION ONLY.

Brief constraint: "the controller must use the estimated state, never ground truth."
This class is the *only* place in the code base that reads the simulator's true
robot pose. It is imported by src/evaluation and main.py's logger, and by the
offline characterisation scripts. It is deliberately not imported anywhere in
src/estimation, src/planning, src/control or src/perception
(tests/test_no_ground_truth_leak.py enforces this).
"""
from __future__ import annotations

import math

import numpy as np


class GroundTruthOracle:
    def __init__(self, robot):
        self._data = robot.data
        self._model = robot.model

    def pose(self) -> np.ndarray:
        """True (x, y, yaw) of the base frame in W."""
        q = self._data.qpos[3:7]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return np.array([self._data.qpos[0], self._data.qpos[1], yaw])

    def velocity(self) -> np.ndarray:
        """True body-frame (v, w)."""
        x, y, yaw = self.pose()
        vx, vy = self._data.qvel[0], self._data.qvel[1]
        return np.array([vx * math.cos(yaw) + vy * math.sin(yaw), self._data.qvel[5]])
