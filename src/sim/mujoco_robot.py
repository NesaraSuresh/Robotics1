"""MuJoCo implementation of the HAL for the TurtleBot3 Waffle Pi.

Sensor models (all noise is added here, MuJoCo itself is noise-free):
  * Wheel encoders: true wheel joint angle quantised to `ticks_per_rev`, plus a
    random-walk increment error (slip / backlash).
  * LiDAR: 360 rays from `lidar_site` via mj_multiRay; Gaussian range noise,
    random dropouts, min/max range.
  * Camera: offscreen render from `rgb_cam` (robot's own visual mesh hidden),
    then Gaussian blur + additive pixel noise.
Actuation: (v, w) -> wheel rates through the inverse differential-drive
kinematics, saturated at the Dynamixel limit, sent to MuJoCo velocity servos.
Disturbances: lateral push via xfrc_applied, camera occlusion, command latency.
"""
from __future__ import annotations

import collections
import math
from typing import Optional

import cv2
import mujoco
import numpy as np

from src.config import Config
from src.hal.interface import CameraImage, EncoderReading, LaserScan, RobotInterface
from src.sim.world import WorldInfo


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class MujocoTurtlebot(RobotInterface):
    def __init__(self, cfg: Config, world: WorldInfo, rng: np.random.Generator,
                 render_camera: bool = True):
        self.cfg = cfg
        self.world = world
        self.rng = rng
        self.model = mujoco.MjModel.from_xml_path(str(world.xml_path))
        self.data = mujoco.MjData(self.model)
        self.ticks_per_rev = cfg.noise.encoder_ticks_per_rev

        m = self.model
        self.base_id = m.body("base").id
        self.jl = m.joint("wheel_left")
        self.jr = m.joint("wheel_right")
        self.lidar_site = m.site("lidar_site").id
        self.n_substeps = int(round(cfg.sim.control_dt / cfg.sim.physics_dt))

        # spawn at the (unknown to the robot) true start pose
        self.data.qpos[0:3] = [world.true_start[0], world.true_start[1], 0.0]
        self.data.qpos[3:7] = yaw_to_quat(world.true_start[2])
        mujoco.mj_forward(m, self.data)

        # camera renderer (robot's own visual mesh group 2 hidden from its camera)
        self.render_camera = render_camera
        self._renderer = None
        if render_camera:
            self._renderer = mujoco.Renderer(m, cfg.sim.cam_height, cfg.sim.cam_width)
            self._cam_opt = mujoco.MjvOption()
            self._cam_opt.geomgroup[2] = 0

        # lidar ray directions in the lidar frame
        n = cfg.sim.lidar_rays
        self.lidar_angles = np.linspace(-math.pi, math.pi, n, endpoint=False)
        self._ray_local = np.stack([np.cos(self.lidar_angles), np.sin(self.lidar_angles),
                                    np.zeros(n)], axis=1)
        self._geomid = np.zeros(n, dtype=np.int32)
        self._dist = np.zeros(n, dtype=np.float64)
        self._geomgroup = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)

        # encoder state (accumulated measurement error per wheel)
        self._enc_err = np.zeros(2)
        self._last_q = np.array([self._wheel_q("l"), self._wheel_q("r")])

        self._step_count = 0
        self._cam_every = max(1, int(round(cfg.sim.camera_period / cfg.sim.control_dt)))
        self._lidar_every = max(1, int(round(cfg.sim.lidar_period / cfg.sim.control_dt)))
        self._cmd_queue = collections.deque([(0.0, 0.0)] * (cfg.disturbance.latency_steps + 1),
                                            maxlen=cfg.disturbance.latency_steps + 1)
        self.last_cmd_wheels = np.zeros(2)

    # ------------------------------------------------------------------ utils
    def _wheel_q(self, side: str) -> float:
        j = self.jl if side == "l" else self.jr
        return float(self.data.qpos[self.model.jnt_qposadr[j.id]])

    def time(self) -> float:
        return float(self.data.time)

    # --------------------------------------------------------------- sensors
    def read_encoders(self) -> EncoderReading:
        q = np.array([self._wheel_q("l"), self._wheel_q("r")])
        dq = q - self._last_q
        self._last_q = q
        # slip/backlash: increment noise proportional to motion + small floor
        std = self.cfg.noise.wheel_increment_std
        self._enc_err += self.rng.normal(0.0, std, 2) * (np.abs(dq) > 1e-4)
        meas = q + self._enc_err
        k = self.ticks_per_rev / (2 * math.pi)
        return EncoderReading(self.time(), int(math.floor(meas[0] * k)), int(math.floor(meas[1] * k)))

    def read_scan(self) -> Optional[LaserScan]:
        if self._step_count % self._lidar_every != 0:
            return None
        m, d, s = self.model, self.data, self.cfg.sim
        R = d.site_xmat[self.lidar_site].reshape(3, 3)
        p = d.site_xpos[self.lidar_site].copy()
        vec = (self._ray_local @ R.T).reshape(-1).copy()
        mujoco.mj_multiRay(m, d, p, vec, self._geomgroup, 1, self.base_id,
                           self._geomid, self._dist, None, s.lidar_rays, s.lidar_range_max)
        r = self._dist.copy()
        r[(self._geomid < 0) | (r < 0)] = np.inf
        noise = self.cfg.noise
        r = r + self.rng.normal(0.0, noise.lidar_range_std, r.shape)
        r[self.rng.random(r.shape) < noise.lidar_dropout] = np.inf
        r[(r < s.lidar_range_min) | (r > s.lidar_range_max)] = np.inf
        return LaserScan(self.time(), self.lidar_angles.copy(), r)

    def read_image(self) -> Optional[CameraImage]:
        if not self.render_camera or self._step_count % self._cam_every != 0:
            return None
        occ = self.cfg.disturbance.occlusion
        if occ is not None and occ[0] <= self.time() <= occ[1]:
            return CameraImage(self.time(), np.zeros((self.cfg.sim.cam_height, self.cfg.sim.cam_width, 3), np.uint8))
        self._renderer.update_scene(self.data, camera="rgb_cam", scene_option=self._cam_opt)
        rgb = self._renderer.render()
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        n = self.cfg.noise
        if n.image_blur_ksize and n.image_blur_ksize > 1:
            img = cv2.GaussianBlur(img, (n.image_blur_ksize, n.image_blur_ksize), 0)
        if n.image_noise_std > 0:
            img = np.clip(img.astype(np.float32) + self.rng.normal(0, n.image_noise_std, img.shape),
                          0, 255).astype(np.uint8)
        return CameraImage(self.time(), img)

    # --------------------------------------------------------------- actuation
    def send_velocity(self, v: float, w: float) -> None:
        self._cmd_queue.append((float(v), float(w)))
        v, w = self._cmd_queue[0]
        rp = self.cfg.robot
        wl = (v - w * rp.wheel_base / 2) / rp.wheel_radius
        wr = (v + w * rp.wheel_base / 2) / rp.wheel_radius
        # saturate preserving curvature
        s = max(1.0, abs(wl) / rp.wheel_rate_max, abs(wr) / rp.wheel_rate_max)
        self.last_cmd_wheels = np.array([wl, wr]) / s
        self.data.ctrl[0], self.data.ctrl[1] = self.last_cmd_wheels

    def step(self) -> None:
        dist = self.cfg.disturbance
        for _ in range(self.n_substeps):
            self.data.xfrc_applied[self.base_id, :] = 0.0
            if dist.push_time is not None and dist.push_time <= self.data.time < dist.push_time + dist.push_duration:
                yaw = self._true_yaw()
                fx, fy = dist.push_force
                self.data.xfrc_applied[self.base_id, 0] = fx * math.cos(yaw) - fy * math.sin(yaw)
                self.data.xfrc_applied[self.base_id, 1] = fx * math.sin(yaw) + fy * math.cos(yaw)
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

    def _true_yaw(self) -> float:
        q = self.data.qpos[3:7]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    # ------------------------------------------------------------ rendering
    def render_view(self, camera: str = "overview", width: int = 640, height: int = 480) -> np.ndarray:
        """Third-person render for videos (not a robot sensor)."""
        if getattr(self, "_view_renderer", None) is None or self._view_renderer.width != width:
            self._view_renderer = mujoco.Renderer(self.model, height, width)
        self._view_renderer.update_scene(self.data, camera=camera)
        return cv2.cvtColor(self._view_renderer.render(), cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        for r in (self._renderer, getattr(self, "_view_renderer", None)):
            if r is not None:
                r.close()
        try:
            self.world.xml_path.unlink()
        except OSError:
            pass
