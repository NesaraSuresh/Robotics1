"""Refresh the numbers / tables on the project page (README.md) from the latest results.

    python scripts/update_readme.py

Copies figures + summaries into docs/, then rewrites every block
<!-- AUTO:NAME -->...<!-- /AUTO:NAME --> in README.md. Run it after
calibrate_camera.py, characterise_noise.py, model_analysis and run_trials.py.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DOCS = ROOT / "docs"


def section(md: str, start: str) -> str:
    """Markdown table that follows a line starting with `start`."""
    lines = md.splitlines()
    out, on = [], False
    for ln in lines:
        if ln.startswith(start):
            on = True
            continue
        if on:
            if ln.startswith("|"):
                out.append(ln)
            elif out:
                break
    return "\n".join(out)


def first_table(md: str) -> str:
    return "\n".join(ln for ln in md.splitlines() if ln.startswith("|"))


def main():
    (DOCS / "figures").mkdir(parents=True, exist_ok=True)
    (DOCS / "results").mkdir(parents=True, exist_ok=True)
    copies = {
        RES / "characterisation" / "camera_noise.png": DOCS / "figures" / "camera_noise.png",
        RES / "characterisation" / "odometry_noise.png": DOCS / "figures" / "odometry_noise.png",
        RES / "calibration" / "views.png": DOCS / "figures" / "calibration_views.png",
        RES / "calibration" / "report.md": DOCS / "calibration_report.md",
        RES / "model_analysis.md": DOCS / "model_analysis.md",
        RES / "nominal_seed1_ekf" / "map.png": DOCS / "figures" / "map_nominal.png",
        RES / "nominal_seed1_ekf" / "estimation.png": DOCS / "figures" / "estimation_nominal.png",
        RES / "nominal_seed1_ekf" / "control.png": DOCS / "figures" / "control_nominal.png",
        RES / "push_seed2_ekf" / "estimation.png": DOCS / "figures" / "estimation_push.png",
        RES / "push_seed2_ekf" / "map.png": DOCS / "figures" / "map_push.png",
    }
    for name in ("robustness", "r_sweep", "lqr_sweep", "ekf_vs_ukf"):
        copies[RES / "trials" / name / "summary.md"] = DOCS / "results" / f"{name}.md"
        copies[RES / "trials" / name / "trials.csv"] = DOCS / "results" / f"{name}_trials.csv"
    for src, dst in copies.items():
        if src.exists():
            shutil.copy(src, dst)

    vals = {}
    cal = (DOCS / "calibration_report.md").read_text() if (DOCS / "calibration_report.md").exists() else ""
    vals["CALIBRATION_TABLE"] = first_table(cal)
    m = re.search(r"mean ([\d.]+) mm, max ([\d.]+) mm over (\d+) views", cal)
    vals["EXTRINSIC_ERR"] = f"{m.group(1)} mm (max {m.group(2)} mm, {m.group(3)} views)" if m else "n/a"
    noise = json.loads((ROOT / "config" / "noise_params.json").read_text())["ekf"]
    vals["SIGMA_B"] = f"{noise['sigma_b0']:.4f}"
    vals["RHO_MAX"] = f"{noise['rho_max_heading']:.1f}"
    vals["A1"] = f"{noise['a1']:.3f}"
    vals["A4"] = f"{noise['a4']:.3f}"
    vals["Q_PARAMS"] = ", ".join(f"{k}={noise[k]:.3g}" for k in ("a1", "a2", "a3", "a4", "v_floor", "w_floor"))
    ma = (DOCS / "model_analysis.md").read_text()
    vals["POLES_TABLE"] = section(ma, "| v_r | w_r | K (2x3)") and ("| v_r | w_r | K (2x3) | closed-loop eigenvalue magnitude (discrete) "
                                                                 "| continuous-equivalent poles [1/s] |\n"
                                                                 + section(ma, "| v_r | w_r | K (2x3)"))
    for key, name in (("ROBUSTNESS_TABLE", "robustness"), ("R_SWEEP_TABLE", "r_sweep"),
                      ("LQR_SWEEP_TABLE", "lqr_sweep"), ("UKF_TABLE", "ekf_vs_ukf")):
        p = DOCS / "results" / f"{name}.md"
        vals[key] = p.read_text().strip() if p.exists() else "_run scripts/run_trials.py to generate_"
    rob = DOCS / "results" / "robustness_trials.csv"
    if rob.exists():
        import csv
        rows = list(csv.DictReader(rob.open()))
        n = max(1, len(rows) // max(1, len({r["scenario"] for r in rows})))
        vals["N_SEEDS"] = str(n)
        vals["CI"] = f"{100 * 1.96 * math.sqrt(0.25 / n):.0f}"

    readme = (ROOT / "README.md").read_text()
    for k, v in vals.items():
        readme = re.sub(rf"<!-- AUTO:{k} -->.*?<!-- /AUTO:{k} -->",
                        lambda _m, k=k, v=v: (f"<!-- AUTO:{k} -->\n{v}\n<!-- /AUTO:{k} -->" if k.endswith("TABLE")
                                              else f"<!-- AUTO:{k} -->{v}<!-- /AUTO:{k} -->"), readme, flags=re.S)
    (ROOT / "README.md").write_text(readme)
    missing = re.findall(r"\{\{(\w+)\}\}", readme)
    print("README updated." + (f" Unfilled: {missing}" if missing else ""))


if __name__ == "__main__":
    sys.exit(main())
