"""One episode of the full pipeline: sense -> estimate -> map/plan -> control -> act.

Data flow per control step (20 Hz):
    encoders --odometry--> (v, w) --EKF.predict-->
    camera (10 Hz) --ArUco+PnP--> z = [r, b, alpha] --EKF.update (gated)-->
    LiDAR (10 Hz) --occupancy grid (at EKF pose)--> A* / safety layer
    mission(EKF pose) --LQR--> (v, w) command --> wheel velocity servos
The ground-truth oracle is read only by the logger (evaluation), after the
command for this step has already been computed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import Config
from src.control.lqr_tracker import LQRTracker
from src.control.mission import DockingMission, Mode
from src.estimation.ekf import LandmarkEKF
from src.estimation.odometry import WheelOdometry
from src.evaluation.logger import EpisodeLog, compute_metrics
from src.perception.aruco_detector import ArucoLandmarkDetector
from src.perception.camera_model import CameraIntrinsics
from src.planning.occupancy_grid import OccupancyGrid
from src.sim.ground_truth import GroundTruthOracle   # evaluation only (see module docstring)
from src.sim.mujoco_robot import MujocoTurtlebot
from src.sim.world import build_world


@dataclass
class EpisodeResult:
    metrics: dict
    log: EpisodeLog
    world: object
    grid: OccupancyGrid


def run_episode(cfg: Config, video_path: Optional[Path] = None, viewer: bool = False,
                verbose: bool = True, estimator_cls=LandmarkEKF) -> EpisodeResult:
    rng_world = np.random.default_rng(cfg.seed)
    rng_sensor = np.random.default_rng(cfg.seed + 10_000)
    world = build_world(cfg, rng_world, tag=f"_{cfg.seed}")
    robot = MujocoTurtlebot(cfg, world, rng_sensor)
    oracle = GroundTruthOracle(robot)

    intr = CameraIntrinsics.load(cfg)
    detector = ArucoLandmarkDetector(intr, {m.id: m.size for m in world.markers})
    landmarks = world.marker_map()
    rp, nz = cfg.robot, cfg.noise
    odom = WheelOdometry(rp.wheel_radius * nz.odom_radius_scale_left, rp.wheel_radius * nz.odom_radius_scale_right,
                         rp.wheel_base * nz.odom_base_scale, robot.ticks_per_rev)
    ekf = estimator_cls(cfg.ekf, world.nominal_start)
    grid = OccupancyGrid(cfg.planner, world.bounds)
    tracker = LQRTracker(cfg.lqr, rp, cfg.sim.control_dt)
    mission = DockingMission(cfg, grid, tracker, world.predock_target, world.dock_target)
    log = EpisodeLog()

    video = None
    if video_path is not None:
        from src.evaluation.video import VideoRecorder
        video = VideoRecorder(video_path, world, robot, grid)
    view = None
    if viewer:
        import mujoco.viewer
        view = mujoco.viewer.launch_passive(robot.model, robot.data)

    last_meas, last_hits = [], np.zeros((0, 2))
    last_img = None
    last_path_id = None
    wall0 = time.time()
    try:
        while robot.time() < cfg.sim.max_time:
            t = robot.time()
            # ---- proprioception + prediction
            v_o, w_o, dt = odom.update(robot.read_encoders())
            ekf.predict(v_o, w_o, dt)
            # ---- exteroception: camera landmarks
            img = robot.read_image()
            if img is not None:
                last_img = img.image
                last_meas = detector.detect(img.image)
                for mm in last_meas:
                    if mm.z[0] > cfg.ekf.max_detection_range:
                        continue
                    info = ekf.update(mm.z, landmarks[mm.id], mm.id)
                    log.updates.append({"t": t, "id": mm.id, "nis": info.nis, "accepted": info.accepted,
                                        "range": float(mm.z[0]), "dim": info.dim})
            # ---- exteroception: lidar mapping
            scan = robot.read_scan()
            if scan is not None:
                last_hits = grid.update(scan, ekf.x)
            # ---- decision + control (estimate only)
            cmd = mission.step(t, ekf.x.copy(), scan)
            robot.send_velocity(cmd.v, cmd.w)
            # ---- evaluation logging (ground truth allowed here only)
            if mission.path is not None and id(mission.path) != last_path_id:
                last_path_id = id(mission.path)
                log.paths.append(mission.path.pts.copy())
            xt = oracle.pose()
            ct = mission.path.cross_track(xt) if (mission.path is not None and cmd.mode == Mode.NAVIGATE) else 0.0
            log.add(t, ekf.x, ekf.P, xt, cmd, len(last_meas) if img is not None else -1, ct)
            if video is not None and img is not None:
                video.frame(t, ekf.x, ekf.P, xt, last_img, last_meas, last_hits, mission, cmd)
            if view is not None:
                view.sync()
                if not view.is_running():
                    break
                time.sleep(max(0.0, cfg.sim.control_dt - (time.time() - wall0)))
                wall0 = time.time()
            if cmd.mode in (Mode.DOCKED, Mode.FAILED):
                break
            robot.step()
    finally:
        if video is not None:
            video.close()
        if view is not None:
            view.close()
    metrics = compute_metrics(log, world.dock_target, cfg, {
        "n_replans": mission.n_replans, "n_safety_stops": mission.n_safety_stops,
        "n_recoveries": ekf.n_recoveries,
        "estimator": estimator_cls.__name__, "intrinsics": intr.source,
    })
    robot.close()
    if verbose:
        print(f"[{cfg.scenario} seed={cfg.seed}] success={metrics['success']} "
              f"dock_err={metrics['dock_pos_err'] * 100:.1f}cm/{metrics['dock_yaw_err_deg']:.1f}deg "
              f"est_rmse={metrics['est_rmse_xy'] * 100:.1f}cm NEES={metrics['nees_mean']:.2f} "
              f"t={metrics['duration']:.1f}s replans={metrics['n_replans']}")
    return EpisodeResult(metrics, log, world, grid)
