"""LQR trajectory tracker for the unicycle, acting on the ESTIMATED state.

Tracking error expressed in the robot frame (Kanayama et al., 1990):
    e1 =  cos(th)(x_r - x) + sin(th)(y_r - y)      along-track
    e2 = -sin(th)(x_r - x) + cos(th)(y_r - y)      cross-track
    e3 = th_r - th                                  heading
Exact error dynamics:
    de1 = w e2 - v + v_r cos e3
    de2 = -w e1 + v_r sin e3
    de3 = w_r - w
With u1 = v_r cos e3 - v, u2 = w_r - w and linearising about e = 0:
    de = A e + B u,  A = [[0, w_r, 0], [-w_r, 0, v_r], [0, 0, 0]],  B = [[1, 0], [0, 0], [0, 1]]
(A, B) is controllable iff v_r != 0 or w_r != 0 (see src/analysis/model_analysis.py).
Discretised with ZOH (first order) at the control period and solved with the
DARE each step (3x3 -> negligible cost), i.e. a gain-scheduled LQR.
Command:  v = v_r cos e3 + K_1 e,   w = w_r + K_2 e   (u = -K e)
Weights by Bryson's rule: Q = diag(1/e_max^2), R = diag(1/u_max^2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are

from src.config import LQRConfig, RobotParams
from src.estimation.ekf import wrap
from src.planning.path import ReferencePoint


def error_dynamics(v_r: float, w_r: float):
    A = np.array([[0.0, w_r, 0.0], [-w_r, 0.0, v_r], [0.0, 0.0, 0.0]])
    B = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    return A, B


def tracking_error(pose: np.ndarray, ref: ReferencePoint) -> np.ndarray:
    c, s = math.cos(pose[2]), math.sin(pose[2])
    dx, dy = ref.x - pose[0], ref.y - pose[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, float(wrap(ref.theta - pose[2]))])


@dataclass
class ControlOutput:
    v: float
    w: float
    e: np.ndarray
    K: np.ndarray


class LQRTracker:
    def __init__(self, cfg: LQRConfig, robot: RobotParams, dt: float):
        self.cfg, self.robot, self.dt = cfg, robot, dt
        self.Q = np.diag(1.0 / np.square(cfg.e_max))
        self.R = np.diag(1.0 / np.square(cfg.u_max))

    def gain(self, v_r: float, w_r: float) -> np.ndarray:
        v_eff = math.copysign(max(abs(v_r), self.cfg.v_ref_min_for_gain), v_r if v_r != 0 else 1.0)
        A, B = error_dynamics(v_eff, w_r)
        Ad = np.eye(3) + A * self.dt
        Bd = B * self.dt
        P = solve_discrete_are(Ad, Bd, self.Q, self.R)
        return np.linalg.solve(self.R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)

    def closed_loop_poles(self, v_r: float, w_r: float) -> np.ndarray:
        A, B = error_dynamics(v_r, w_r)
        K = self.gain(v_r, w_r)
        return np.linalg.eigvals(np.eye(3) + A * self.dt - B * self.dt @ K)

    def control(self, pose_est: np.ndarray, ref: ReferencePoint) -> ControlOutput:
        e = tracking_error(pose_est, ref)
        K = self.gain(ref.v, ref.w)
        u = -K @ e
        v = ref.v * math.cos(e[2]) - u[0]
        w = ref.w - u[1]
        return ControlOutput(*self.saturate(v, w), e, K)

    def rotate_to(self, pose_est: np.ndarray, heading: float) -> ControlOutput:
        """In-place rotation (v_r = 0: the LQR model is uncontrollable here, so a
        separate heading loop handles it). Proportional on wrapped heading error."""
        e3 = float(wrap(heading - pose_est[2]))
        w = self.cfg.k_heading * e3
        w = math.copysign(max(abs(w), 0.15), w) if abs(e3) > 0.02 else 0.0
        return ControlOutput(*self.saturate(0.0, w), np.array([0.0, 0.0, e3]), np.zeros((2, 3)))

    def saturate(self, v: float, w: float):
        rp = self.robot
        v = float(np.clip(v, -rp.v_max, rp.v_max))
        w = float(np.clip(w, -rp.w_max, rp.w_max))
        return v, w
