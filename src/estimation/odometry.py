"""Wheel-encoder odometry: ticks -> body-frame (v, w) over one control period.

Inverse of the differential-drive kinematics used by the actuator:
    v = (r_R * dphi_R + r_L * dphi_L) / (2 dt)
    w = (r_R * dphi_R - r_L * dphi_L) / (L dt)
r_L, r_R, L are the odometry's *believed* parameters; if they differ from the
true robot (the "slip" scenario) odometry drifts systematically.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from src.hal.interface import EncoderReading


class WheelOdometry:
    def __init__(self, r_left: float, r_right: float, wheel_base: float, ticks_per_rev: int):
        self.rl, self.rr, self.L = r_left, r_right, wheel_base
        self.k = 2 * math.pi / ticks_per_rev
        self._last: Optional[EncoderReading] = None

    def update(self, enc: EncoderReading) -> Tuple[float, float, float]:
        """Returns (v, w, dt). First call returns zeros."""
        if self._last is None:
            self._last = enc
            return 0.0, 0.0, 0.0
        dt = enc.t - self._last.t
        dl = (enc.ticks_left - self._last.ticks_left) * self.k
        dr = (enc.ticks_right - self._last.ticks_right) * self.k
        self._last = enc
        if dt <= 0:
            return 0.0, 0.0, 0.0
        ds = (self.rr * dr + self.rl * dl) / 2
        dth = (self.rr * dr - self.rl * dl) / self.L
        return ds / dt, dth / dt, dt
