"""Result figures for the project page (matplotlib, headless)."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle
import numpy as np
from scipy.stats import chi2

from src.estimation.ekf import wrap

C_TRUE, C_EST, C_PATH, C_OBS = "#1b9e77", "#d95f02", "#7570b3", "#555555"
MODE_COLORS = {"NAVIGATE": "#e8f1fb", "ALIGN_PATH": "#fdf2e0", "ALIGN_DOCK": "#fdf2e0", "DOCK": "#eaf7ea"}


def _cov_ellipse(ax, mean, P2, nsig=2.0, **kw):
    vals, vecs = np.linalg.eigh(P2)
    vals = np.maximum(vals, 1e-12)
    ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(mean, 2 * nsig * math.sqrt(vals[1]), 2 * nsig * math.sqrt(vals[0]), angle=ang,
                         fill=False, **kw))


def _shade_modes(ax, t, modes):
    start = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or modes[i] != modes[start]:
            c = MODE_COLORS.get(modes[start])
            if c:
                ax.axvspan(t[start], t[min(i, len(t) - 1)], color=c, lw=0, zorder=0)
            start = i


def plot_map(res, out: Path, title: str = "") -> None:
    a = res.log.arrays()
    w = res.world
    fig, ax = plt.subplots(figsize=(8, 6.6))
    g = res.grid
    occ = g.occupied()
    ext = [g.x0, g.x0 + g.nx * g.res, g.y0, g.y0 + g.ny * g.res]
    ax.imshow(np.where(occ, 0.25, 1.0), origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1, alpha=0.55)
    for o in w.obstacles:   # true obstacle outlines (evaluation overlay)
        if o.kind == "cylinder":
            ax.add_patch(Circle(o.pos, o.size[0], fill=False, ec=C_OBS, ls="--", lw=0.8))
        else:
            hx, hy = o.size
            c, s = math.cos(o.yaw), math.sin(o.yaw)
            corner = np.array(o.pos) - np.array([c * hx - s * hy, s * hx + c * hy])
            ax.add_patch(Rectangle(corner, 2 * hx, 2 * hy, angle=math.degrees(o.yaw), fill=False, ec=C_OBS,
                                   ls="--", lw=0.8))
    for i, p in enumerate(res.log.paths):
        ax.plot(p[:, 0], p[:, 1], color=C_PATH, lw=1.0, alpha=0.35 if i < len(res.log.paths) - 1 else 0.9,
                label="planned path (A*)" if i == len(res.log.paths) - 1 else None)
    ax.plot(a["x_true"][:, 0], a["x_true"][:, 1], color=C_TRUE, lw=2, label="ground truth")
    ax.plot(a["x_est"][:, 0], a["x_est"][:, 1], color=C_EST, lw=1.2, ls="--", label="EKF estimate")
    for k in range(0, len(a["t"]), 40):
        _cov_ellipse(ax, a["x_est"][k, :2], a["P"][k][:2, :2], 2.0, ec=C_EST, lw=0.8, alpha=0.8)
    for m in w.markers:
        n = m.normal
        ax.plot(m.pos[0], m.pos[1], "ks", ms=6)
        ax.annotate(f"{'DOCK' if m.id == w.dock_marker.id else 'id'} {m.id}", m.pos[:2] + 0.18 * n[:2],
                    ha="center", fontsize=8)
    ax.plot(*w.dock_target[:2], "*", color="gold", ms=14, mec="k", label="dock target")
    ax.plot(*w.nominal_start[:2], "o", color="w", mec="k", label="nominal start")
    b = w.bounds
    ax.set_xlim(b[0] - 0.2, b[1] + 0.2)
    ax.set_ylim(b[2] - 0.2, b[3] + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title or "Trajectory, LiDAR occupancy map, 2-sigma EKF ellipses")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_estimation(res, out: Path) -> None:
    a = res.log.arrays()
    t = a["t"]
    err = a["x_est"] - a["x_true"]
    err[:, 2] = wrap(err[:, 2])
    sig = np.sqrt(np.array([np.diag(P) for P in a["P"]]))
    fig, axs = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    labels = ["x error [m]", "y error [m]", "yaw error [deg]"]
    for i in range(3):
        k = 180 / math.pi if i == 2 else 1.0
        ax = axs[i]
        _shade_modes(ax, t, a["mode"])
        ax.fill_between(t, -3 * sig[:, i] * k, 3 * sig[:, i] * k, color=C_EST, alpha=0.18, lw=0, label="$\\pm3\\sigma$")
        ax.plot(t, err[:, i] * k, color=C_EST, lw=1.0, label="estimate - truth")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(labels[i])
    axs[0].legend(loc="upper right", fontsize=8)
    # marker visibility rug
    vis = a["n_visible"]
    for i in range(3):
        y0 = axs[i].get_ylim()[0]
        axs[i].plot(t[vis > 0], np.full((vis > 0).sum(), y0), "|", color="k", ms=4, alpha=0.4)
    P = a["P"]
    nees = np.array([e @ np.linalg.solve(Pk, e) for e, Pk in zip(err, P)])
    ax = axs[3]
    _shade_modes(ax, t, a["mode"])
    ax.semilogy(t, np.maximum(nees, 1e-3), color="#444", lw=0.8, label="NEES")
    ax.axhline(3, color="g", lw=1, label="E[NEES]=3")
    ax.axhline(chi2.ppf(0.975, 3), color="r", ls="--", lw=0.8, label="95% bounds")
    ax.axhline(chi2.ppf(0.025, 3), color="r", ls="--", lw=0.8)
    ax.set_ylabel("NEES")
    ax.set_xlabel("time [s]  (background: blue = NAVIGATE, green = DOCK; ticks = marker visible)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    fig.suptitle("EKF error with 3-sigma bounds and consistency (NEES)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_control(res, out: Path) -> None:
    a = res.log.arrays()
    t = a["t"]
    fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax in axs:
        _shade_modes(ax, t, a["mode"])
    axs[0].plot(t, a["u"][:, 0], label="v cmd", color="#1f77b4")
    axs[0].plot(t, a["u_ref"][:, 0], "--", label="v ref", color="#1f77b4", alpha=0.6)
    axs[0].set_ylabel("v [m/s]")
    axs[0].legend(fontsize=8)
    axs[1].plot(t, a["u"][:, 1], label="w cmd", color="#ff7f0e")
    axs[1].plot(t, a["u_ref"][:, 1], "--", label="w ref", color="#ff7f0e", alpha=0.6)
    axs[1].set_ylabel("w [rad/s]")
    axs[1].legend(fontsize=8)
    axs[2].plot(t, a["e_track"][:, 1] * 100, label="cross-track e2 (estimated) [cm]")
    axs[2].plot(t, a["cross_track_true"] * 100, label="distance to path (true) [cm]", alpha=0.7)
    axs[2].plot(t, np.degrees(a["e_track"][:, 2]), label="heading e3 [deg]", alpha=0.7)
    axs[2].set_ylabel("tracking error")
    axs[2].set_xlabel("time [s]")
    axs[2].legend(fontsize=8)
    fig.suptitle("LQR tracking: commands, feed-forward reference and tracking error")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_all(res, out_dir: Path, title: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_map(res, out_dir / "map.png", title)
    plot_estimation(res, out_dir / "estimation.png")
    plot_control(res, out_dir / "control.png")
