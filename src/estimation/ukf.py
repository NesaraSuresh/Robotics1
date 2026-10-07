"""Unscented Kalman Filter with the same interface as LandmarkEKF (HD extension:
"an additional estimator compared rigorously").

Same process/measurement models and noise as the EKF (inherits them), but the
nonlinear propagation uses 2n+1 sigma points (scaled UT, Wan & van der Merwe)
instead of Jacobians. Angles are averaged with circular means and residuals
are wrapped. Process noise enters through the inputs by augmenting the state
with the input noise (n = 3 + 2 = 5) so the UKF and EKF see identical noise.
"""
from __future__ import annotations

import math

import numpy as np

from src.estimation.ekf import LandmarkEKF, UpdateInfo, wrap


class LandmarkUKF(LandmarkEKF):
    alpha, beta, kappa = 0.5, 2.0, 0.0

    def _weights(self, n: int):
        lam = self.alpha ** 2 * (n + self.kappa) - n
        wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = wm[0] + (1 - self.alpha ** 2 + self.beta)
        return lam, wm, wc

    def _sigma(self, mean: np.ndarray, cov: np.ndarray):
        n = len(mean)
        lam, wm, wc = self._weights(n)
        L = np.linalg.cholesky((n + lam) * (cov + 1e-12 * np.eye(n)))
        pts = np.vstack([mean, mean + L.T, mean - L.T])
        return pts, wm, wc

    @staticmethod
    def _state_mean(X: np.ndarray, wm: np.ndarray) -> np.ndarray:
        m = wm @ X
        m[2] = math.atan2(wm @ np.sin(X[:, 2]), wm @ np.cos(X[:, 2]))
        return m

    def predict(self, v: float, w: float, dt: float) -> None:
        if dt <= 0:
            return
        M = self.cfg.q_scale * self.input_noise(v, w)
        xa = np.concatenate([self.x, [0.0, 0.0]])
        Pa = np.zeros((5, 5))
        Pa[:3, :3] = self.P
        Pa[3:, 3:] = M
        S, wm, wc = self._sigma(xa, Pa)
        X = np.array([self.motion(s[:3], v + s[3], w + s[4], dt) for s in S])
        mean = self._state_mean(X, wm)
        d = X - mean
        d[:, 2] = wrap(d[:, 2])
        self.x = mean
        self.P = (wc[:, None] * d).T @ d + np.diag(self.cfg.q_add)
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z_full: np.ndarray, lm: np.ndarray, marker_id: int = -1) -> UpdateInfo:
        dim = self.measurement_dim(z_full[0], lm[3])
        z = np.asarray(z_full[:dim], dtype=float)
        X, wm, wc = self._sigma(self.x, self.P)
        Z = np.array([self.h(s, lm, dim) for s in X])
        zhat = wm @ Z
        for k in range(1, dim):
            zhat[k] = math.atan2(wm @ np.sin(Z[:, k]), wm @ np.cos(Z[:, k]))
        dz = Z - zhat
        dz[:, 1:] = wrap(dz[:, 1:])
        dx = X - self.x
        dx[:, 2] = wrap(dx[:, 2])
        Rk = self.R(z[0], lm[3], dim)
        S = (wc[:, None] * dz).T @ dz + Rk
        Pxz = (wc[:, None] * dx).T @ dz
        y = z - zhat
        y[1:] = wrap(y[1:])
        Sinv = np.linalg.inv(S)
        nis = float(y @ Sinv @ y)
        if nis > self.gates[dim]:
            self._gate_bookkeeping(False, nis, self.gates[dim])
            return UpdateInfo(marker_id, False, nis, self.gates[dim], y, S)
        K = Pxz @ Sinv
        self.x = self.x + K @ y
        self.x[2] = float(wrap(self.x[2]))
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._gate_bookkeeping(True)
        return UpdateInfo(marker_id, True, nis, self.gates[dim], y, S)
