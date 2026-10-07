"""Pinhole camera model and the fixed camera->base extrinsic.

Frames (see docs/frames.md):
  B  base_link  : origin mid-axle on the floor, x forward, y left, z up
  C  camera_optical (OpenCV): origin at lens, z forward, x right, y down
MuJoCo's own camera frame is x right, y up, z backwards; the optical frame used
by OpenCV is that frame rotated 180 deg about x. We never use the MuJoCo camera
frame downstream -- only C.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from src.config import ROOT, Config

# camera lens position in B (from assets/tb3/turtlebot3_waffle_pi.xml, camera rgb_cam)
CAM_POS_B = np.array([0.0865, 0.0, 0.107])
# columns = C axes expressed in B
R_B_C = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]])


def T_B_C() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_B_C
    T[:3, 3] = CAM_POS_B
    return T


@dataclass
class CameraIntrinsics:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int
    source: str

    @staticmethod
    def analytic(cfg: Config) -> "CameraIntrinsics":
        """Ideal pinhole from MuJoCo's vertical field of view (square pixels)."""
        W, H = cfg.sim.cam_width, cfg.sim.cam_height
        f = (H / 2) / math.tan(math.radians(cfg.sim.cam_fovy_deg) / 2)
        K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])
        return CameraIntrinsics(K, np.zeros(5), W, H, "analytic (fovy)")

    @staticmethod
    def load(cfg: Config) -> "CameraIntrinsics":
        """Calibrated intrinsics if scripts/calibrate_camera.py has been run, else analytic."""
        p = ROOT / "config" / "camera_calibration.json"
        if p.exists():
            d = json.loads(p.read_text())
            if d["width"] == cfg.sim.cam_width and d["height"] == cfg.sim.cam_height:
                return CameraIntrinsics(np.array(d["K"]), np.array(d["dist"]), d["width"], d["height"],
                                        "checkerboard calibration")
        return CameraIntrinsics.analytic(cfg)
