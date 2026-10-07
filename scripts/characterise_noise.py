"""Identify the EKF noise models from data (Q from odometry, R from the camera).

This is an OFFLINE calibration experiment, so it is allowed to use ground truth
(the same way you would use a motion-capture system on the real robot).

Camera R: the robot is placed at a grid of poses relative to a marker (range,
viewing angle, image bearing offset), images are rendered *with* the sensor noise
model, and z = [r, b, alpha] from ArUco+PnP is compared with the true value.
Errors are binned by rho = range / marker_size (apparent size) and fitted:
    sigma_r = r0 + kr rho^2,  sigma_b = b0 + kb rho,  sigma_alpha = a0 + ka rho
using a robust (MAD) standard deviation per bin; the alpha outlier rate decides
rho_max_heading.

Odometry Q: the robot drives a set of constant (v, w) commands; per control step
the odometry (v, w) is compared to the true body velocity and the std of the
error is regressed on |v| and |w| -> (a1, a2, a3, a4).

Writes config/noise_params.json (read automatically by src/config.make_config)
and figures to results/characterisation/.
    python scripts/characterise_noise.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src  # noqa: F401  (selects the OpenGL backend before mujoco is imported)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.optimize import nnls

from src.config import RESULTS, ROOT, make_config
from src.estimation.ekf import wrap
from src.estimation.odometry import WheelOdometry
from src.perception.aruco_detector import ArucoLandmarkDetector
from src.perception.camera_model import CameraIntrinsics
from src.sim.ground_truth import GroundTruthOracle
from src.sim.mujoco_robot import MujocoTurtlebot, yaw_to_quat
from src.sim.world import build_world


def robust_std(x):
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def camera_experiment(quick: bool):
    cfg = make_config("nominal", seed=0, overrides={"world": {"n_obstacles": 0}})
    rng = np.random.default_rng(0)
    world = build_world(cfg, rng, tag="_char")
    robot = MujocoTurtlebot(cfg, world, rng)
    det = ArucoLandmarkDetector(CameraIntrinsics.load(cfg), {m.id: m.size for m in world.markers})
    oracle = GroundTruthOracle(robot)
    rows = []
    targets = [world.dock_marker] + [m for m in world.markers if m.id == 1]
    dists = np.linspace(0.3, 4.0, 8 if quick else 16)
    views = np.radians([-50, -25, 0, 25, 50] if quick else [-60, -45, -30, -15, 0, 15, 30, 45, 60])
    offs = np.radians([-15, 0, 15])
    reps = 2 if quick else 3
    for mk in targets:
        for d in dists:
            for phi in views:
                for off in offs:
                    yaw_n = mk.yaw + phi
                    x = mk.pos[0] + d * math.cos(yaw_n)
                    y = mk.pos[1] + d * math.sin(yaw_n)
                    if not (-2.9 < x < 2.9 and -2.4 < y < 2.4):
                        continue
                    heading = math.atan2(mk.pos[1] - y, mk.pos[0] - x) + off
                    robot.data.qpos[:3] = [x, y, 0.0]
                    robot.data.qpos[3:7] = yaw_to_quat(heading)
                    robot.data.qvel[:] = 0
                    mujoco.mj_forward(robot.model, robot.data)
                    gt = oracle.pose()
                    dx, dy = mk.pos[0] - gt[0], mk.pos[1] - gt[1]
                    z_true = np.array([math.hypot(dx, dy), wrap(math.atan2(dy, dx) - gt[2]), wrap(mk.yaw - gt[2])])
                    for _ in range(reps):
                        robot._step_count = 0
                        img = robot.read_image()
                        meas = [m for m in det.detect(img.image) if m.id == mk.id]
                        detected = len(meas) > 0
                        e = (meas[0].z - z_true) if detected else np.full(3, np.nan)
                        if detected:
                            e[1:] = wrap(e[1:])
                        rows.append(dict(id=mk.id, size=mk.size, r=z_true[0], rho=z_true[0] / mk.size,
                                         view=math.degrees(phi), detected=detected,
                                         er=e[0], eb=e[1], ea=e[2]))
    robot.close()
    return rows


def fit_camera(rows, out_dir: Path):
    det = [r for r in rows if r["detected"]]
    rho = np.array([r["rho"] for r in det])
    E = np.array([[r["er"], r["eb"], r["ea"]] for r in det])
    edges = np.array([0, 3, 5, 7.5, 10, 13, 17, 22, 28, 36])
    centres, sd, out_rate, det_rate = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (rho >= lo) & (rho < hi)
        allsel = [r for r in rows if lo <= r["rho"] < hi]
        if sel.sum() < 5:
            continue
        centres.append(rho[sel].mean())
        s = np.array([robust_std(E[sel, k]) for k in range(3)])
        sd.append(s)
        out_rate.append(np.mean(np.abs(E[sel, 2]) > max(3 * s[2], np.radians(5))))
        det_rate.append(np.mean([r["detected"] for r in allsel]))
    c, sd = np.array(centres), np.array(sd)
    # non-negative least squares for each model
    r0, kr = nnls(np.stack([np.ones_like(c), c ** 2], 1), sd[:, 0])[0]
    b0, kb = nnls(np.stack([np.ones_like(c), c], 1), sd[:, 1])[0]
    a0, ka = nnls(np.stack([np.ones_like(c), c], 1), sd[:, 2])[0]
    out_rate = np.array(out_rate)
    # largest bin centre before the alpha outlier rate first exceeds 5 %
    first_bad = int(np.argmax(out_rate > 0.05)) if np.any(out_rate > 0.05) else len(c)
    rho_max = float(c[max(first_bad - 1, 0)])
    params = dict(sigma_r0=float(max(r0, 1e-3)), sigma_kr=float(kr), sigma_b0=float(max(b0, 5e-4)),
                  sigma_kb=float(kb), sigma_a0=float(max(a0, 2e-3)), sigma_ka=float(ka),
                  rho_max_heading=rho_max)
    # ---- figure
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
    xx = np.linspace(0, c.max() * 1.05, 100)
    names = ["range error [m]", "bearing error [rad]", "relative-heading error [rad]"]
    models = [r0 + kr * xx ** 2, b0 + kb * xx, a0 + ka * xx]
    for k in range(3):
        axs[k].plot(rho, np.abs(E[:, k]), ".", ms=2, alpha=0.3, color="grey", label="|error| samples")
        axs[k].plot(c, sd[:, k], "o", color="C1", label="robust std per bin")
        axs[k].plot(xx, models[k], "-", color="C0", label="fitted sigma model")
        axs[k].plot(xx, 3 * models[k], ":", color="C0", label="3 sigma")
        axs[k].set_yscale("log")
        axs[k].set_xlabel("rho = range / marker size")
        axs[k].set_title(names[k])
    axs[0].legend(fontsize=7)
    axs[2].axvline(rho_max, color="r", ls="--", lw=1)
    axs[3].plot(c, out_rate, "o-", label="alpha outlier rate")
    axs[3].plot(c, det_rate, "s-", label="detection rate")
    axs[3].axvline(rho_max, color="r", ls="--", lw=1, label=f"rho_max_heading={rho_max:.1f}")
    axs[3].set_xlabel("rho")
    axs[3].legend(fontsize=7)
    axs[3].set_title("PnP ambiguity + detection")
    fig.tight_layout()
    fig.savefig(out_dir / "camera_noise.png", dpi=140)
    plt.close(fig)
    return params, dict(rho_bins=c.tolist(), std=sd.tolist(), alpha_outlier_rate=out_rate.tolist(),
                        detection_rate=det_rate, n_samples=len(rows), n_detected=len(det))


def odometry_experiment(quick: bool, window: int = 20):
    """Drive constant (v, w) commands; compare odometry with truth over 1 s windows.

    Per-step differences of the simulator's true pose contain high-frequency contact
    jitter that averages out, so the per-step noise is identified from errors
    integrated over `window` steps:  sigma_step = std(window-average error) * sqrt(window).
    """
    cfg = make_config("nominal", seed=0, overrides={"world": {"n_obstacles": 0, "start_nominal": (0.0, 0.0, 0.0),
                                                              "start_sigma": (0, 0, 0)}})
    rng = np.random.default_rng(1)
    world = build_world(cfg, rng, tag="_odo")
    robot = MujocoTurtlebot(cfg, world, rng, render_camera=False)
    oracle = GroundTruthOracle(robot)
    rp = cfg.robot
    odom = WheelOdometry(rp.wheel_radius, rp.wheel_radius, rp.wheel_base, robot.ticks_per_rev)
    cmds = [(v, w) for v in (0.0, 0.08, 0.16, 0.24) for w in (-1.2, -0.5, 0.0, 0.5, 1.2) if (v, w) != (0, 0)]
    if quick:
        cmds = cmds[::2]
    rows = []
    n_steps = 10 + window * (4 if quick else 8)
    for v_c, w_c in cmds:
        robot.data.qpos[:3] = [0, 0, 0]
        robot.data.qpos[3:7] = yaw_to_quat(0)
        robot.data.qvel[:] = 0
        mujoco.mj_forward(robot.model, robot.data)
        odom.update(robot.read_encoders())
        acc = None
        for k in range(n_steps):
            robot.send_velocity(v_c, w_c)
            robot.step()
            v_o, w_o, dt = odom.update(robot.read_encoders())
            if k < 10:          # skip the acceleration transient
                continue
            if acc is None:
                acc = dict(p0=oracle.pose(), ds=0.0, dth=0.0, n=0, T=0.0, path=0.0, prev=oracle.pose())
                continue
            cur = oracle.pose()
            acc["ds"] += v_o * dt
            acc["dth"] += w_o * dt
            acc["path"] += float(np.dot(cur[:2] - acc["prev"][:2],
                                        [math.cos(acc["prev"][2]), math.sin(acc["prev"][2])]))
            acc["prev"] = cur
            acc["n"] += 1
            acc["T"] += dt
            if acc["n"] == window:
                dth_true = wrap(cur[2] - acc["p0"][2])
                rows.append((acc["ds"] / acc["T"], acc["dth"] / acc["T"],
                             (acc["ds"] - acc["path"]) / acc["T"], wrap(acc["dth"] - dth_true) / acc["T"]))
                acc = dict(p0=cur, ds=0.0, dth=0.0, n=0, T=0.0, path=0.0, prev=cur)
    robot.close()
    return np.array(rows), window


def fit_odometry(rows_window, out_dir: Path):
    rows, window = rows_window
    v, w, ev, ew = rows.T
    key = np.round(np.stack([np.abs(v), np.abs(w)], 1), 1)
    cells = {}
    for k_, a, b in zip(map(tuple, key), ev, ew):
        cells.setdefault(k_, []).append((a, b))
    X, tv, tw, bias_v, bias_w = [], [], [], [], []
    for (av, aw), vals in cells.items():
        vals = np.array(vals)
        if len(vals) < 3:
            continue
        X.append([av, aw, 1.0])
        sv = np.std(vals[:, 0]) * math.sqrt(window)      # per-step equivalent
        sw = np.std(vals[:, 1]) * math.sqrt(window)
        bv, bw = np.mean(vals[:, 0]), np.mean(vals[:, 1])  # systematic slip
        bias_v.append(float(bv))
        bias_w.append(float(bw))
        tv.append(math.sqrt(sv ** 2 + bv ** 2))
        tw.append(math.sqrt(sw ** 2 + bw ** 2))
    X, tv, tw = np.array(X), np.array(tv), np.array(tw)
    a1, a2, vf = nnls(X, tv)[0]
    a3, a4, wf = nnls(X, tw)[0]
    params = dict(a1=float(a1), a2=float(a2), v_floor=float(max(vf, 1e-3)),
                  a3=float(a3), a4=float(a4), w_floor=float(max(wf, 2e-3)))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    axs[0].plot(X[:, 0] + 0.003 * X[:, 1], tv, "o", label="identified (std + |bias|)")
    axs[0].plot(X[:, 0] + 0.003 * X[:, 1], X @ [a1, a2, vf], "x", label="model a1|v|+a2|w|+c")
    axs[0].set_xlabel("|v| [m/s]")
    axs[0].set_ylabel("sigma_v per step [m/s]")
    axs[0].legend(fontsize=8)
    axs[1].plot(X[:, 1] + 0.01 * X[:, 0], tw, "o", label="identified (std + |bias|)")
    axs[1].plot(X[:, 1] + 0.01 * X[:, 0], X @ [a3, a4, wf], "x", label="model a3|v|+a4|w|+c")
    axs[1].set_xlabel("|w| [rad/s]")
    axs[1].set_ylabel("sigma_w per step [rad/s]")
    axs[1].legend(fontsize=8)
    fig.suptitle(f"Odometry noise identified over {window}-step (1 s) windows")
    fig.tight_layout()
    fig.savefig(out_dir / "odometry_noise.png", dpi=140)
    plt.close(fig)
    return params, dict(slip_bias_v=bias_v, slip_bias_w=bias_w, window_steps=window)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out_dir = RESULTS / "characterisation"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("camera experiment ...")
    cam_rows = camera_experiment(args.quick)
    cam_params, cam_diag = fit_camera(cam_rows, out_dir)
    print("  ", {k: round(v, 5) for k, v in cam_params.items()})
    print("odometry experiment ...")
    odo_rows = odometry_experiment(args.quick)
    odo_params, odo_diag = fit_odometry(odo_rows, out_dir)
    print("  ", {k: round(v, 5) for k, v in odo_params.items()})
    out = {"ekf": {**cam_params, **odo_params},
           "diagnostics": {"camera": cam_diag, "odometry": odo_diag}}
    (ROOT / "config").mkdir(exist_ok=True)
    (ROOT / "config" / "noise_params.json").write_text(json.dumps(out, indent=2))
    print("wrote config/noise_params.json and results/characterisation/*.png")


if __name__ == "__main__":
    main()
