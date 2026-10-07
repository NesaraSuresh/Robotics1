"""Episode logging + metrics. Ground truth enters the pipeline ONLY here.

Metrics (all computed against ground truth, after the run):
  success            final |p - p_dock| < success_pos_tol and |yaw err| < success_yaw_tol
  dock_pos_err/yaw   final docking error
  est_rmse_*         RMSE of (estimate - truth) over the episode
  nees_mean          mean NEES (3 dof); consistent filter -> ~3
  nees_in_95         fraction of steps with NEES inside the 95% chi2(3) interval
  nis_mean           mean NIS of accepted updates (dof 2 or 3)
  cross_track_rmse   true distance from the planned path while NAVIGATE (estimation + control)
  track_e2/e3_rmse   controller's own cross-track / heading error w.r.t. its estimate (control only)
  effort             mean(v^2/v_max^2 + w^2/w_max^2)  (dimensionless control effort)
  time_to_dock, n_updates, n_rejected, n_replans, n_safety_stops
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import chi2

from src.estimation.ekf import wrap


@dataclass
class EpisodeLog:
    t: List[float] = field(default_factory=list)
    x_est: List[np.ndarray] = field(default_factory=list)
    P: List[np.ndarray] = field(default_factory=list)
    x_true: List[np.ndarray] = field(default_factory=list)
    u: List[tuple] = field(default_factory=list)            # (v_cmd, w_cmd)
    u_ref: List[tuple] = field(default_factory=list)
    e_track: List[np.ndarray] = field(default_factory=list)  # tracking error (estimated)
    cross_track_true: List[float] = field(default_factory=list)
    mode: List[str] = field(default_factory=list)
    n_visible: List[int] = field(default_factory=list)
    safety: List[bool] = field(default_factory=list)
    updates: List[dict] = field(default_factory=list)        # per EKF update
    paths: List[np.ndarray] = field(default_factory=list)    # every (re)planned path

    def add(self, t, x_est, P, x_true, cmd, n_visible, cross_track):
        self.t.append(t)
        self.x_est.append(np.array(x_est))
        self.P.append(np.array(P))
        self.x_true.append(np.array(x_true))
        self.u.append((cmd.v, cmd.w))
        self.u_ref.append((cmd.v_ref, cmd.w_ref))
        self.e_track.append(np.array(cmd.e))
        self.mode.append(cmd.mode.value)
        self.n_visible.append(n_visible)
        self.safety.append(cmd.safety_stop)
        self.cross_track_true.append(cross_track)

    def arrays(self) -> Dict[str, np.ndarray]:
        return {
            "t": np.array(self.t), "x_est": np.array(self.x_est), "P": np.array(self.P),
            "x_true": np.array(self.x_true), "u": np.array(self.u), "u_ref": np.array(self.u_ref),
            "e_track": np.array(self.e_track), "mode": np.array(self.mode),
            "n_visible": np.array(self.n_visible), "safety": np.array(self.safety),
            "cross_track_true": np.array(self.cross_track_true),
        }

    def save(self, path: Path) -> None:
        a = self.arrays()
        upd = self.updates
        np.savez_compressed(path, **a,
                            upd_t=np.array([u["t"] for u in upd]), upd_id=np.array([u["id"] for u in upd]),
                            upd_nis=np.array([u["nis"] for u in upd]),
                            upd_acc=np.array([u["accepted"] for u in upd]),
                            upd_range=np.array([u["range"] for u in upd]),
                            upd_dim=np.array([u["dim"] for u in upd]))


def compute_metrics(log: EpisodeLog, dock_target: np.ndarray, cfg, extra: dict) -> dict:
    a = log.arrays()
    err = a["x_est"] - a["x_true"]
    err[:, 2] = wrap(err[:, 2])
    nees = np.array([e @ np.linalg.solve(P, e) for e, P in zip(err, a["P"])])
    lo, hi = chi2.ppf(0.025, 3), chi2.ppf(0.975, 3)
    final = a["x_true"][-1]
    pos_err = float(np.linalg.norm(final[:2] - dock_target[:2]))
    yaw_err = float(abs(wrap(final[2] - dock_target[2])))
    m = cfg.mission
    docked_flag = a["mode"][-1] == "DOCKED"
    success = bool(docked_flag and pos_err < m.success_pos_tol and math.degrees(yaw_err) < m.success_yaw_tol_deg)
    acc = [u for u in log.updates if u["accepted"]]
    nav = a["mode"] == "NAVIGATE"
    rp = cfg.robot
    effort = float(np.mean((a["u"][:, 0] / rp.v_max) ** 2 + (a["u"][:, 1] / rp.w_max) ** 2))
    t_dock = float(a["t"][-1]) if docked_flag else float("nan")
    return {
        "scenario": cfg.scenario, "seed": cfg.seed, "success": success, "docked": bool(docked_flag),
        "dock_pos_err": pos_err, "dock_yaw_err_deg": math.degrees(yaw_err),
        "final_est_err": float(np.linalg.norm(err[-1, :2])),
        "est_rmse_xy": float(np.sqrt(np.mean(np.sum(err[:, :2] ** 2, axis=1)))),
        "est_rmse_yaw_deg": float(np.degrees(np.sqrt(np.mean(err[:, 2] ** 2)))),
        "est_max_xy": float(np.max(np.linalg.norm(err[:, :2], axis=1))),
        "nees_mean": float(np.mean(nees)), "nees_in_95": float(np.mean((nees > lo) & (nees < hi))),
        "nis_mean": float(np.mean([u["nis"] for u in acc])) if acc else float("nan"),
        "n_updates": len(acc), "n_rejected": len(log.updates) - len(acc),
        "cross_track_rmse": float(np.sqrt(np.mean(a["cross_track_true"][nav] ** 2))) if nav.any() else float("nan"),
        "track_e2_rmse": float(np.sqrt(np.mean(a["e_track"][nav, 1] ** 2))) if nav.any() else float("nan"),
        "track_e3_rmse_deg": float(np.degrees(np.sqrt(np.mean(a["e_track"][nav, 2] ** 2)))) if nav.any() else float("nan"),
        "effort": effort, "time_to_dock": t_dock, "duration": float(a["t"][-1]),
        **extra,
    }


def save_metrics(metrics: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2, default=float))
