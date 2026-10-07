"""Monte-Carlo evaluation: repeated trials over scenarios / seeds / parameter sweeps.

Examples
    # robustness table: every scenario, 10 random worlds each
    python scripts/run_trials.py --scenarios nominal noisy slip push occlusion stress --seeds 10

    # R-sweep (estimator consistency vs responsiveness trade-off)
    python scripts/run_trials.py --scenarios nominal --seeds 6 --sweep ekf.r_scale=0.1,1,10,100

    # LQR weight sweep (control effort vs tracking error trade-off)
    python scripts/run_trials.py --scenarios nominal --seeds 6 --sweep lqr.u_max="(0.05,0.3)","(0.1,0.6)","(0.2,1.2)"

    # EKF vs UKF
    python scripts/run_trials.py --scenarios nominal stress --seeds 10 --estimators ekf ukf

Writes results/trials/<name>/trials.csv, summary.csv and summary.md.
Seeds are paired across conditions (same worlds), so differences are not luck.
"""
from __future__ import annotations

import argparse
import ast
import csv
import itertools
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src  # noqa: F401,E402

import numpy as np  # noqa: E402

from src.config import RESULTS, make_config  # noqa: E402

KEYS = ["success", "dock_pos_err", "dock_yaw_err_deg", "est_rmse_xy", "est_rmse_yaw_deg", "est_max_xy",
        "nees_mean", "nees_in_95", "nis_mean", "cross_track_rmse", "track_e2_rmse", "track_e3_rmse_deg", "effort", "duration", "n_rejected",
        "n_recoveries", "n_replans", "n_safety_stops"]


def _one(job):
    scenario, seed, estimator, overrides, label = job
    from src.runner import run_episode
    if estimator == "ukf":
        from src.estimation.ukf import LandmarkUKF as est
    else:
        from src.estimation.ekf import LandmarkEKF as est
    cfg = make_config(scenario, seed, overrides)
    try:
        m = run_episode(cfg, verbose=False, estimator_cls=est).metrics
    except Exception as exc:  # a crash is a failed trial, keep going
        m = {"success": False, "error": repr(exc)}
    m.update(scenario=scenario, seed=seed, estimator=estimator, condition=label)
    return m


def parse_sweep(spec: str):
    key, vals = spec.split("=", 1)
    section, name = key.split(".")
    out = []
    parts, depth, cur = [], 0, ""
    for ch in vals:                      # split on commas outside parentheses
        depth += ch in "([" 
        depth -= ch in ")]"
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    for v in parts:
        val = ast.literal_eval(v)
        out.append((f"{key}={v}", {section: {name: val}}))
    return out


def summarise(rows, group_keys):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)
    summary = []
    for g, rs in groups.items():
        s = dict(zip(group_keys, g))
        s["n"] = len(rs)
        s["success_rate"] = float(np.mean([bool(r.get("success")) for r in rs]))
        for k in KEYS[1:]:
            vals = np.array([r.get(k, np.nan) for r in rs], dtype=float)
            vals = vals[np.isfinite(vals)]
            s[k + "_mean"] = float(vals.mean()) if len(vals) else float("nan")
            s[k + "_std"] = float(vals.std()) if len(vals) else float("nan")
        summary.append(s)
    return summary


def to_markdown(summary, group_keys) -> str:
    cols = group_keys + ["n", "success_rate", "dock_pos_err", "dock_yaw_err_deg", "est_rmse_xy", "nees_mean",
                         "track_e2_rmse", "cross_track_rmse", "effort", "duration"]
    head = ["condition" if c == "condition" else c for c in cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(cols)]
    for s in summary:
        cells = []
        for c in cols:
            if c in group_keys or c == "n":
                cells.append(str(s[c]))
            elif c == "success_rate":
                cells.append(f"{100 * s[c]:.0f} %")
            else:
                m, sd = s[c + "_mean"], s[c + "_std"]
                scale = 100 if c in ("dock_pos_err", "est_rmse_xy", "cross_track_rmse", "track_e2_rmse") else 1
                unit = " cm" if scale == 100 else ""
                cells.append(f"{m * scale:.2f} ± {sd * scale:.2f}{unit}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=["nominal"])
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (1..N)")
    ap.add_argument("--seed-offset", type=int, default=100)
    ap.add_argument("--estimators", nargs="+", default=["ekf"], choices=["ekf", "ukf"])
    ap.add_argument("--sweep", default=None, help='e.g. ekf.r_scale=0.1,1,10')
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    conditions = parse_sweep(args.sweep) if args.sweep else [("default", None)]
    jobs = [(sc, args.seed_offset + k, est, ov, label)
            for (label, ov), sc, est, k in itertools.product(conditions, args.scenarios, args.estimators,
                                                             range(args.seeds))]
    name = args.name or time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS / "trials" / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} trials on {args.workers} workers -> {out}")
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            print(f"  [{i}/{len(jobs)}] {r['condition']:<22} {r['scenario']:<10} seed={r['seed']} {r['estimator']} "
                  f"success={r.get('success')} dock={r.get('dock_pos_err', math.nan) * 100:.1f}cm "
                  f"NEES={r.get('nees_mean', math.nan):.2f}  ({time.time() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: (r["condition"], r["scenario"], r["estimator"], r["seed"]))
    fields = sorted({k for r in rows for k in r})
    with open(out / "trials.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    group_keys = ["condition", "scenario", "estimator"]
    summary = summarise(rows, group_keys)
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    md = to_markdown(summary, group_keys)
    (out / "summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
