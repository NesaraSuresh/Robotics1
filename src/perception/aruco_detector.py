"""ArUco detection + PnP -> landmark measurements in the robot base frame.

Pipeline per image:
  1. cv2.aruco detection (DICT_4X4_50, sub-pixel corner refinement)
  2. keep ids that are in the known landmark map
  3. solvePnP(IPPE_SQUARE) -> T_C_M (marker pose in the optical frame)
  4. T_B_M = T_B_C @ T_C_M
  5. z = [range, bearing, relative_heading] of the marker in B, where
        range   = || p_B,xy ||
        bearing = atan2(p_y, p_x)
        alpha   = atan2(n_y, n_x)  (direction of the marker normal in B)
     The EKF's measurement model predicts exactly these from (x, y, theta) and
     the marker's known world pose (estimation/ekf.py, h()).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np

from src.perception.camera_model import CameraIntrinsics, T_B_C
from src.sim.world import ARUCO_DICT


@dataclass
class MarkerMeasurement:
    id: int
    z: np.ndarray               # [range, bearing, alpha]
    corners: np.ndarray         # 4x2 pixels (for display)
    reproj_err: float           # [px]
    T_B_M: np.ndarray           # 4x4


class ArucoLandmarkDetector:
    def __init__(self, intrinsics: CameraIntrinsics, marker_sizes: Dict[int, float]):
        self.cam = intrinsics
        self.sizes = marker_sizes
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 3
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)
        self.T_B_C = T_B_C()

    @staticmethod
    def object_points(size: float) -> np.ndarray:
        s = size / 2
        # OpenCV corner order: top-left, top-right, bottom-right, bottom-left (marker frame, y up)
        return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)

    def detect(self, image_bgr: np.ndarray, max_reproj_px: float = 2.0) -> List[MarkerMeasurement]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        out: List[MarkerMeasurement] = []
        if ids is None:
            return out
        for c, mid in zip(corners, ids.flatten()):
            mid = int(mid)
            if mid not in self.sizes:
                continue
            img_pts = c.reshape(4, 2).astype(np.float64)
            obj = self.object_points(self.sizes[mid])
            ok, rvec, tvec = cv2.solvePnP(obj, img_pts, self.cam.K, self.cam.dist,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            proj, _ = cv2.projectPoints(obj, rvec, tvec, self.cam.K, self.cam.dist)
            err = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - img_pts) ** 2, axis=1))))
            if err > max_reproj_px:
                continue
            T_C_M = np.eye(4)
            T_C_M[:3, :3] = cv2.Rodrigues(rvec)[0]
            T_C_M[:3, 3] = tvec.ravel()
            T_B_M = self.T_B_C @ T_C_M
            p = T_B_M[:3, 3]
            n = T_B_M[:3, 2]                      # marker z axis = outward normal
            if np.dot(n[:2], p[:2]) > 0:          # normal must face the robot
                continue
            z = np.array([math.hypot(p[0], p[1]), math.atan2(p[1], p[0]), math.atan2(n[1], n[0])])
            out.append(MarkerMeasurement(mid, z, img_pts, err, T_B_M))
        return out

    @staticmethod
    def draw(image_bgr: np.ndarray, meas: List[MarkerMeasurement]) -> np.ndarray:
        img = image_bgr.copy()
        for m in meas:
            pts = m.corners.astype(int)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            cv2.putText(img, f"id{m.id} r={m.z[0]:.2f}m", tuple(pts[0] + np.array([0, -6])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return img
