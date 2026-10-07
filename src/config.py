"""Central configuration for the G82 Track A pipeline.

Every tunable number lives here so the project page can quote it and the viva
can point at it. Values marked (*) are justified in README.md / docs.
Scenario presets at the bottom are used by main.py and scripts/run_trials.py.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"


@dataclass
class RobotParams:
    """TurtleBot3 Waffle Pi (ROBOTIS e-Manual)."""
    wheel_radius: float = 0.033        # r [m]
    wheel_base: float = 0.288          # L [m], wheel separation
    v_max: float = 0.26                # [m/s] e-Manual max linear speed
    w_max: float = 1.82                # [rad/s] e-Manual max angular speed
    wheel_rate_max: float = 7.88       # [rad/s] actuator ctrlrange
    front_offset: float = 0.0865       # base origin -> front (camera lens) [m]
    radius: float = 0.21               # footprint circumscribed radius [m]


@dataclass
class SimConfig:
    physics_dt: float = 0.002
    control_dt: float = 0.05           # 20 Hz control / odometry / EKF predict
    camera_period: float = 0.10        # 10 Hz camera
    lidar_period: float = 0.10         # 10 Hz LiDAR (LDS-01 is 5 Hz; 10 Hz chosen for sim)
    cam_width: int = 640
    cam_height: int = 480
    cam_fovy_deg: float = 48.8         # Pi Camera v2 vertical FoV
    lidar_rays: int = 360
    lidar_range_min: float = 0.12
    lidar_range_max: float = 3.5
    max_time: float = 150.0            # episode timeout [s]


@dataclass
class NoiseConfig:
    """Sensor noise + model mismatch injected by the simulator HAL.

    MuJoCo is noise-free, so all realism is added here. The estimator never
    reads these values directly; its own noise model is in EKFConfig and is
    identified with scripts/characterise_noise.py.
    """
    encoder_ticks_per_rev: int = 4096          # Dynamixel XM430 resolution
    wheel_increment_std: float = 0.002         # [rad] per control step, slip/backlash
    odom_radius_scale_left: float = 1.00       # odometry's belief of r (calibration error)
    odom_radius_scale_right: float = 1.00
    odom_base_scale: float = 1.00              # odometry's belief of L
    lidar_range_std: float = 0.01              # [m] LDS-01 spec ~ +/-10 mm (<0.5 m) ... 3.5 %
    lidar_dropout: float = 0.02                # fraction of returns lost
    image_noise_std: float = 4.0               # [grey levels] additive Gaussian
    image_blur_ksize: int = 3                  # Gaussian blur kernel (0 = off)


@dataclass
class EKFConfig:
    # Odometry (prediction) noise: sigma_v = a1|v| + a2|w| + v_floor etc.  (*)
    a1: float = 0.05
    a2: float = 0.01
    a3: float = 0.02
    a4: float = 0.10
    v_floor: float = 0.002
    w_floor: float = 0.005
    q_add: tuple = (1e-6, 1e-6, 1e-6)          # small additive process noise per step
    # Camera measurement noise as a function of rho = range / marker_size (apparent size).  (*)
    #   sigma_r = r0 + kr rho^2 ,  sigma_b = b0 + kb rho ,  sigma_alpha = a0 + ka rho
    # Defaults are overwritten by config/noise_params.json (scripts/characterise_noise.py).
    sigma_r0: float = 0.003
    sigma_kr: float = 0.0004
    sigma_b0: float = 0.002
    sigma_kb: float = 0.0003
    sigma_a0: float = 0.01
    sigma_ka: float = 0.006
    rho_max_heading: float = 10.0              # use alpha only when the marker is big enough
    use_rel_heading: bool = True               # z = [r, b, alpha] (pose) vs [r, b]
    gate_prob: float = 0.997                   # chi-square gate on NIS
    r_scale: float = 10.0                      # R inflation for dynamic effects (body rock, timing) not seen in the
                                               # static characterisation; chosen by the R-sweep: NEES ~ 3 (README)
    q_scale: float = 1.0                       # global Q multiplier
    max_consecutive_rejects: int = 8           # gate lock-out recovery: inflate P after this many
    recovery_sigma: tuple = (0.10, 0.10, 0.08)
    recovery_mode: str = "fixed"               # "fixed" | "adaptive" (P scaled by NIS/gate, see ekf.py)
    recovery_max_scale: float = 400.0          # cap on the adaptive scale factor
    p0_sigma: tuple = (0.15, 0.15, 0.10)       # initial std [m, m, rad]
    max_detection_range: float = 3.5           # ignore detections further than this


@dataclass
class LQRConfig:
    # Bryson's rule: Q = diag(1/e_max^2), R = diag(1/u_max^2)  (*)
    e_max: tuple = (0.10, 0.05, 0.10)          # along, cross [m], heading [rad]
    u_max: tuple = (0.10, 0.60)                # dv [m/s], dw [rad/s]
    v_ref_min_for_gain: float = 0.03           # avoids the uncontrollable v_r = 0 linearisation
    k_heading: float = 2.0                     # in-place rotation P-gain [1/s]


@dataclass
class PlannerConfig:
    resolution: float = 0.05
    inflation: float = 0.26                    # robot radius 0.21 + 0.05 margin
    replan_period: float = 1.0
    max_deviation: float = 0.35                # replan if estimate is this far from path
    l_occ: float = 0.85                        # log-odds increments
    l_free: float = -0.4
    l_min: float = -2.0
    l_max: float = 3.5
    occ_threshold: float = 0.7
    smooth_iterations: int = 60


@dataclass
class MissionConfig:
    v_nav: float = 0.18                        # cruise speed [m/s]
    a_lat_max: float = 0.25                    # [m/s^2] for curvature speed limit
    a_long_max: float = 0.20
    predock_distance: float = 0.85             # base origin to marker plane at pre-dock
    dock_distance: float = 0.35                # base origin to marker plane when docked
    v_dock_max: float = 0.08
    v_dock_min: float = 0.025
    k_dock_speed: float = 0.4
    align_tolerance: float = 0.06              # [rad]
    predock_tolerance: float = 0.12            # [m]
    safety_distance: float = 0.12              # LiDAR stop if an obstacle is this close to the robot front [m]
    safety_half_angle_deg: float = 35.0
    success_pos_tol: float = 0.05              # evaluation thresholds (ground truth)
    success_yaw_tol_deg: float = 10.0


@dataclass
class WorldConfig:
    room_x: float = 6.0
    room_y: float = 5.0
    wall_height: float = 0.5
    n_obstacles: int = 5
    randomise_obstacles: bool = True
    start_nominal: tuple = (-2.2, -1.6, 0.6)   # what the robot *believes* its start pose is
    start_sigma: tuple = (0.10, 0.10, 0.08)    # true start = nominal + N(0, sigma)
    dock_y: float = 0.0                        # dock on +x wall at this y
    dock_marker_id: int = 0
    dock_marker_size: float = 0.12
    dock_marker_height: float = 0.15
    landmark_marker_size: float = 0.20
    landmark_marker_height: float = 0.22
    use_landmarks: bool = True                 # extra wall ArUco landmarks (ids 1..)


@dataclass
class DisturbanceConfig:
    push_time: Optional[float] = None          # [s] apply lateral push at this time
    push_force: tuple = (0.0, 25.0)            # body-frame force [N]
    push_duration: float = 0.25
    occlusion: Optional[tuple] = None          # (t_start, t_end) camera blackout
    latency_steps: int = 0                     # command delay in control steps


@dataclass
class Config:
    robot: RobotParams = field(default_factory=RobotParams)
    sim: SimConfig = field(default_factory=SimConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    ekf: EKFConfig = field(default_factory=EKFConfig)
    lqr: LQRConfig = field(default_factory=LQRConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    seed: int = 0
    scenario: str = "nominal"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def _apply(cfg: Config, overrides: dict) -> Config:
    for section, values in overrides.items():
        target = getattr(cfg, section)
        for k, v in values.items():
            if not hasattr(target, k):
                raise KeyError(f"unknown config key {section}.{k}")
            setattr(target, k, v)
    return cfg


SCENARIOS = {
    # Nominal: default noise, randomised start + obstacles.
    "nominal": {},
    # Heavier sensor noise.
    "noisy": {
        "noise": {"wheel_increment_std": 0.006, "lidar_range_std": 0.03,
                  "image_noise_std": 10.0, "image_blur_ksize": 5},
    },
    # Odometry calibration error (wrong wheel radius / base) -> systematic drift.
    "slip": {
        "noise": {"odom_radius_scale_left": 0.97, "odom_radius_scale_right": 1.02,
                  "odom_base_scale": 1.05, "wheel_increment_std": 0.004},
    },
    # Lateral shove mid-mission (unobservable by odometry).
    "push": {"disturbance": {"push_time": 12.0}},
    # Camera blackout for 10 s while navigating.
    "occlusion": {"disturbance": {"occlusion": (5.0, 15.0)}},
    # Everything at once.
    "stress": {
        "noise": {"wheel_increment_std": 0.006, "image_noise_std": 8.0,
                  "odom_radius_scale_left": 0.98, "odom_base_scale": 1.03},
        "disturbance": {"push_time": 12.0, "occlusion": (20.0, 26.0), "latency_steps": 1},
    },
    # Ablation: odometry only (no camera updates) -> shows unobservability / drift.
    "odom_only": {"ekf": {"max_detection_range": 0.0}},
}


def make_config(scenario: str = "nominal", seed: int = 0, overrides: Optional[dict] = None) -> Config:
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown scenario '{scenario}', choose from {list(SCENARIOS)}")
    cfg = Config(seed=seed, scenario=scenario)
    _apply(cfg, copy.deepcopy(SCENARIOS[scenario]))
    noise_file = ROOT / "config" / "noise_params.json"
    if noise_file.exists():  # written by scripts/characterise_noise.py
        identified = json.loads(noise_file.read_text())
        _apply(cfg, {"ekf": identified.get("ekf", {})})
    if overrides:
        _apply(cfg, overrides)
    return cfg
