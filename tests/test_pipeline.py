"""Unit tests:  python -m pytest -q"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

import src  # noqa: F401
from src.config import make_config
from src.control.lqr_tracker import LQRTracker, error_dynamics
from src.estimation.ekf import LandmarkEKF, wrap
from src.estimation.ukf import LandmarkUKF
from src.planning.astar import astar

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ constraint
FORBIDDEN = {"GroundTruthOracle", "ground_truth", "true_start", "x_true"}
DECISION_PACKAGES = ["estimation", "planning", "control", "perception"]


def test_no_ground_truth_in_decision_code():
    """Brief: the controller must use the estimated state, never ground truth."""
    for pkg in DECISION_PACKAGES:
        for f in (ROOT / "src" / pkg).glob("*.py"):
            tree = ast.parse(f.read_text())
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            mods = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in n.names} | {getattr(n, "module", "") or "" for n in ast.walk(tree)
                                         if isinstance(n, ast.ImportFrom)}
            leaked = (names | {m.split(".")[-1] for m in mods}) & FORBIDDEN
            assert not leaked, f"{f.name} references ground truth: {leaked}"
            assert "qpos" not in f.read_text(), f"{f.name} reads simulator state directly"


# ------------------------------------------------------------------ EKF
@pytest.fixture
def ekf():
    cfg = make_config()
    return LandmarkEKF(cfg.ekf, np.array([0.3, -0.2, 0.4]))


def numeric_jac(f, x, eps=1e-6):
    f0 = f(x)
    J = np.zeros((len(f0), len(x)))
    for i in range(len(x)):
        dx = np.zeros(len(x))
        dx[i] = eps
        d = f(x + dx) - f0
        d[1:] = wrap(d[1:]) if len(d) > 1 else d[1:]
        J[:, i] = d / eps
    return J


def test_measurement_jacobian(ekf):
    lm = np.array([2.0, 1.0, math.pi, 0.2])
    x = np.array([0.3, -0.2, 0.4])
    for dim in (2, 3):
        assert np.allclose(ekf.H(x, lm, dim), numeric_jac(lambda s: ekf.h(s, lm, dim), x), atol=1e-5)


def test_motion_jacobian(ekf):
    x = np.array([0.3, -0.2, 0.4])
    F, G = ekf.jacobians(x, 0.2, 0.5, 0.05)
    Fn = numeric_jac(lambda s: ekf.motion(s, 0.2, 0.5, 0.05), x)
    Gn = numeric_jac(lambda u: ekf.motion(x, u[0], u[1], 0.05), np.array([0.2, 0.5]))
    assert np.allclose(F, Fn, atol=1e-5) and np.allclose(G, Gn, atol=1e-5)


@pytest.mark.parametrize("cls", [LandmarkEKF, LandmarkUKF])
def test_filter_converges_and_is_consistent(cls):
    """Synthetic run with exactly the modelled noise: error -> small, NEES ~ 3."""
    cfg = make_config(overrides={"ekf": {"r_scale": 1.0}})
    rng = np.random.default_rng(0)
    lms = [np.array([2.0, 1.0, math.pi, 0.2]), np.array([-1.0, 2.0, -math.pi / 2, 0.2])]
    x_true = np.array([0.0, 0.0, 0.0])
    f = cls(cfg.ekf, x_true + rng.normal(0, cfg.ekf.p0_sigma))
    nees = []
    for k in range(400):
        v, w = 0.15, 0.3 * math.sin(k / 40)
        x_true = f.motion(x_true, v, w, 0.05)
        M = f.input_noise(v, w)
        vn, wn = v + rng.normal(0, math.sqrt(M[0, 0])), w + rng.normal(0, math.sqrt(M[1, 1]))
        f.predict(vn, wn, 0.05)
        if k % 2 == 0:
            for lm in lms:
                z = f.h(x_true, lm, 3)
                R = f.R(z[0], lm[3], 3)
                z = z + rng.normal(0, np.sqrt(np.diag(R)))
                f.update(z, lm)
        e = f.x - x_true
        e[2] = wrap(e[2])
        nees.append(e @ np.linalg.solve(f.P, e))
    assert np.linalg.norm(e[:2]) < 0.05
    assert 1.0 < np.mean(nees[50:]) < 6.0


def test_gate_rejects_outlier(ekf):
    lm = np.array([2.0, 1.0, math.pi, 0.2])
    ekf.P = np.diag([1e-4, 1e-4, 1e-4])
    z = ekf.h(ekf.x, lm) + np.array([1.0, 0.0, 0.0])   # 1 m range outlier
    assert not ekf.update(z, lm).accepted


# ------------------------------------------------------------------ control
def test_error_model_controllability():
    for vr, wr, expect in [(0, 0, 2), (0.2, 0, 3), (0, 0.5, 3)]:
        A, B = error_dynamics(vr, wr)
        C = np.hstack([B, A @ B, A @ A @ B])
        assert np.linalg.matrix_rank(C) == expect


def test_lqr_closed_loop_stable():
    cfg = make_config()
    tr = LQRTracker(cfg.lqr, cfg.robot, cfg.sim.control_dt)
    for vr, wr in [(0.05, 0), (0.18, 0), (0.18, 0.6), (0.1, -0.8)]:
        assert np.all(np.abs(tr.closed_loop_poles(vr, wr)) < 1.0)


# ------------------------------------------------------------------ planning
def test_astar_goes_around_wall():
    grid = np.zeros((40, 40), bool)
    grid[5:35, 20] = True
    path = astar(grid, (20, 5), (20, 35))
    assert path is not None and path[0] == (20, 5) and path[-1] == (20, 35)
    assert not any(grid[r, c] for r, c in path)


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_gate_lockout_recovery(mode):
    """After an unmodelled 'kidnap' (estimate 10 deg / 0.3 m off with small P) the filter
    must eventually re-accept measurements; adaptive recovery should need fewer rounds."""
    cfg = make_config(overrides={"ekf": {"r_scale": 1.0, "recovery_mode": mode}})
    lms = [np.array([2.0, 1.0, math.pi, 0.2]), np.array([2.0, -1.0, math.pi, 0.2])]
    x_true = np.array([0.0, 0.0, 0.0])
    f = LandmarkEKF(cfg.ekf, np.array([0.3, -0.2, math.radians(10)]))
    f.P = np.diag([1e-4, 1e-4, 1e-5])
    for k in range(200):
        z = f.h(x_true, lms[k % 2], 3)
        if f.update(z, lms[k % 2]).accepted:
            break
    assert k < 199, "filter never re-acquired"
    if mode == "adaptive":
        assert f.n_recoveries <= 2
