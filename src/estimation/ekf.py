"""Extended Kalman Filter: wheel odometry (predict) + ArUco landmark (update).

State      x = [x, y, theta]^T            (base frame B in world W)
Input      u = [v, w]^T                   (from wheel odometry)
Landmark   m = [m_x, m_y, m_theta]        (known marker pose in W, from the map)
Measurement z = [r, b, alpha]^T  (alpha optional)
    r     = sqrt(dx^2 + dy^2)             dx = m_x - x, dy = m_y - y
    b     = atan2(dy, dx) - theta
    alpha = m_theta - theta               (direction of the marker normal in B)

Process model (mid-point integration over dt):
    x+ = x + v dt cos(theta + w dt/2)
    y+ = y + v dt sin(theta + w dt/2)
    th+ = theta + w dt
Process noise is injected through the inputs (velocity-motion model,
Thrun et al. Probabilistic Robotics, Ch. 5):
    M = diag( (a1|v| + a2|w| + v_floor)^2 , (a3|v| + a4|w| + w_floor)^2 )
    Q = G M G^T + Q_add
Measurement noise depends on the marker's apparent size, rho = r / marker_size
(identified in scripts/characterise_noise.py):
    sigma_r = r0 + kr rho^2,  sigma_b = b0 + kb rho,  sigma_alpha = a0 + ka rho
alpha (PnP relative heading) is only used when rho < rho_max_heading: for small
markers the planar-PnP solution is ambiguous and alpha can be off by many degrees.
Updates are gated on the NIS with a chi-square test and use the Joseph form.
Gate lock-out recovery: if `max_consecutive_rejects` updates in a row fail the gate
the filter is assumed overconfident and P is inflated (logged as a recovery event).
  recovery_mode = "fixed":    P <- P + diag(recovery_sigma^2)          (same step every time)
  recovery_mode = "adaptive": P <- f P + diag(recovery_sigma^2),  f = clip(1.2 * min(NIS) / gate, 1, f_max)
    min(NIS) over the rejected run is the smallest surprise seen. Scaling P by NIS/gate makes
    that innovation just pass the gate (if R is small next to HPH^T), so a large unmodelled
    disturbance (push) gets a large widening and a marginal one a small widening. Using the
    *smallest* NIS stops a single wild outlier from blowing P up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from src.config import EKFConfig


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class UpdateInfo:
    marker_id: int
    accepted: bool
    nis: float
    gate: float
    innovation: np.ndarray
    S: np.ndarray

    @property
    def dim(self) -> int:
        return len(self.innovation)


class LandmarkEKF:
    def __init__(self, cfg: EKFConfig, x0: np.ndarray):
        self.cfg = cfg
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.diag(np.square(cfg.p0_sigma)).astype(float)
        self.dim_z = 3 if cfg.use_rel_heading else 2
        self.gates = {d: float(chi2.ppf(cfg.gate_prob, d)) for d in (2, 3)}
        self.gate = self.gates[self.dim_z]
        self.consecutive_rejects = 0
        self.n_recoveries = 0
        self._rejected_nis = []

    # ------------------------------------------------------------- predict
    @staticmethod
    def motion(x: np.ndarray, v: float, w: float, dt: float) -> np.ndarray:
        th_m = x[2] + 0.5 * w * dt
        return np.array([x[0] + v * dt * math.cos(th_m), x[1] + v * dt * math.sin(th_m),
                         float(wrap(x[2] + w * dt))])

    @staticmethod
    def jacobians(x: np.ndarray, v: float, w: float, dt: float):
        th_m = x[2] + 0.5 * w * dt
        c, s = math.cos(th_m), math.sin(th_m)
        F = np.array([[1.0, 0.0, -v * dt * s],
                      [0.0, 1.0, v * dt * c],
                      [0.0, 0.0, 1.0]])
        G = np.array([[dt * c, -0.5 * v * dt * dt * s],
                      [dt * s, 0.5 * v * dt * dt * c],
                      [0.0, dt]])
        return F, G

    def input_noise(self, v: float, w: float) -> np.ndarray:
        c = self.cfg
        sv = c.a1 * abs(v) + c.a2 * abs(w) + c.v_floor
        sw = c.a3 * abs(v) + c.a4 * abs(w) + c.w_floor
        return np.diag([sv * sv, sw * sw])

    def predict(self, v: float, w: float, dt: float) -> None:
        if dt <= 0:
            return
        F, G = self.jacobians(self.x, v, w, dt)
        Q = self.cfg.q_scale * (G @ self.input_noise(v, w) @ G.T) + np.diag(self.cfg.q_add)
        self.x = self.motion(self.x, v, w, dt)
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    # -------------------------------------------------------------- update
    def h(self, x: np.ndarray, lm: np.ndarray, dim: int = 3) -> np.ndarray:
        dx, dy = lm[0] - x[0], lm[1] - x[1]
        z = [math.hypot(dx, dy), float(wrap(math.atan2(dy, dx) - x[2]))]
        if dim == 3:
            z.append(float(wrap(lm[2] - x[2])))
        return np.array(z)

    def H(self, x: np.ndarray, lm: np.ndarray, dim: int = 3) -> np.ndarray:
        dx, dy = lm[0] - x[0], lm[1] - x[1]
        q = dx * dx + dy * dy
        r = math.sqrt(q)
        rows = [[-dx / r, -dy / r, 0.0], [dy / q, -dx / q, -1.0]]
        if dim == 3:
            rows.append([0.0, 0.0, -1.0])
        return np.array(rows)

    def R(self, r: float, size: float, dim: int = 3) -> np.ndarray:
        c = self.cfg
        rho = r / size
        diag = [(c.sigma_r0 + c.sigma_kr * rho * rho) ** 2, (c.sigma_b0 + c.sigma_kb * rho) ** 2]
        if dim == 3:
            diag.append((c.sigma_a0 + c.sigma_ka * rho) ** 2)
        return c.r_scale * np.diag(diag)

    def measurement_dim(self, r: float, size: float) -> int:
        return 3 if (self.cfg.use_rel_heading and r / size < self.cfg.rho_max_heading) else 2

    def _gate_bookkeeping(self, accepted: bool, nis: float = 0.0, gate: float = 1.0) -> None:
        if accepted:
            self.consecutive_rejects = 0
            self._rejected_nis = []
            return
        self.consecutive_rejects += 1
        self._rejected_nis.append(nis / gate)
        if self.consecutive_rejects >= self.cfg.max_consecutive_rejects:
            add = np.diag(np.square(self.cfg.recovery_sigma))
            if self.cfg.recovery_mode == "adaptive":
                f = float(np.clip(1.2 * min(self._rejected_nis), 1.0, self.cfg.recovery_max_scale))
                self.P = f * self.P + add
            else:
                self.P = self.P + add
            self.consecutive_rejects = 0
            self._rejected_nis = []
            self.n_recoveries += 1

    def update(self, z_full: np.ndarray, lm: np.ndarray, marker_id: int = -1) -> UpdateInfo:
        """lm = [m_x, m_y, m_theta, marker_size]."""
        dim = self.measurement_dim(z_full[0], lm[3])
        z = np.asarray(z_full[:dim], dtype=float)
        zhat = self.h(self.x, lm, dim)
        Hk = self.H(self.x, lm, dim)
        Rk = self.R(z[0], lm[3], dim)
        y = z - zhat
        y[1:] = wrap(y[1:])
        S = Hk @ self.P @ Hk.T + Rk
        Sinv = np.linalg.inv(S)
        nis = float(y @ Sinv @ y)
        if nis > self.gates[dim]:
            self._gate_bookkeeping(False, nis, self.gates[dim])
            return UpdateInfo(marker_id, False, nis, self.gates[dim], y, S)
        K = self.P @ Hk.T @ Sinv
        self.x = self.x + K @ y
        self.x[2] = float(wrap(self.x[2]))
        I_KH = np.eye(3) - K @ Hk
        self.P = I_KH @ self.P @ I_KH.T + K @ Rk @ K.T     # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        self._gate_bookkeeping(True)
        return UpdateInfo(marker_id, True, nis, self.gates[dim], y, S)
