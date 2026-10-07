"""G82 Track A demo -- one command runs the full pipeline.

    python main.py                              # nominal run, figures in results/
    python main.py --scenario push --seed 3     # disturbed trial
    python main.py --record                     # also write the demo video (mp4)
    python main.py --viewer                     # live MuJoCo viewer (needs a display)

Scenarios: nominal, noisy, slip, push, occlusion, stress, odom_only (see src/config.py).
"""
from __future__ import annotations

import argparse
import time

from src.config import RESULTS, SCENARIOS, make_config
from src.evaluation.logger import save_metrics
from src.evaluation.plots import plot_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="nominal", choices=list(SCENARIOS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--record", action="store_true", help="write results/<run>/demo.mp4")
    ap.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer")
    ap.add_argument("--estimator", default="ekf", choices=["ekf", "ukf"])
    ap.add_argument("--fixed-world", action="store_true", help="use the fixed obstacle layout")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    overrides = {"world": {"randomise_obstacles": False}} if args.fixed_world else None
    cfg = make_config(args.scenario, args.seed, overrides)
    out = RESULTS / (args.out or f"{args.scenario}_seed{args.seed}_{args.estimator}")
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")

    from src.runner import run_episode
    est_cls = None
    if args.estimator == "ukf":
        from src.estimation.ukf import LandmarkUKF as est_cls
    else:
        from src.estimation.ekf import LandmarkEKF as est_cls

    t0 = time.time()
    res = run_episode(cfg, video_path=(out / "demo.mp4") if args.record else None, viewer=args.viewer,
                      estimator_cls=est_cls)
    save_metrics(res.metrics, out / "metrics.json")
    res.log.save(out / "log.npz")
    plot_all(res, out, title=f"{args.scenario}, seed {args.seed}: "
                             f"{'DOCKED' if res.metrics['success'] else 'FAILED'} "
                             f"({res.metrics['dock_pos_err'] * 100:.1f} cm)")
    print(f"results in {out}  (wall time {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
