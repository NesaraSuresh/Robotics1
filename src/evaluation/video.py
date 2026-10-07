"""Demo-video recorder (brief: show the estimate and its uncertainty alongside the true state).

Frame layout (1280x720):
  left   : third-person MuJoCo render
  right-top   : robot camera with ArUco detections
  right-bottom: top-down map -- occupancy grid, path, LiDAR hits, TRUE pose (green),
                EKF pose (orange) with its 2-sigma ellipse, mode + errors.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from src.estimation.ekf import wrap


class VideoRecorder:
    W, H = 1280, 720

    def __init__(self, path: Path, world, robot, grid, fps: float = 20.0):
        self.path = Path(path)
        self.world, self.robot, self.grid = world, robot, grid
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.W, self.H))
        self.true_trail, self.est_trail = [], []
        b = world.bounds
        self.map_w, self.map_h = 640, 360
        self.scale = min((self.map_w - 20) / (b[1] - b[0]), (self.map_h - 50) / (b[3] - b[2]))
        self.ox = (self.map_w - (b[1] - b[0]) * self.scale) / 2 - b[0] * self.scale
        self.oy = 26 + b[3] * self.scale

    def _px(self, x, y):
        return int(self.ox + x * self.scale), int(self.oy - y * self.scale)

    def _map_panel(self, t, x_est, P, x_true, hits, mission, cmd):
        img = np.full((self.map_h, self.map_w, 3), 255, np.uint8)
        g = self.grid
        occ = g.occupied()
        rr, cc = np.nonzero(occ)
        xs, ys = g.to_world(rr, cc)
        for x, y in zip(xs, ys):
            p = self._px(x, y)
            cv2.rectangle(img, p, (p[0] + max(1, int(g.res * self.scale)), p[1] + max(1, int(g.res * self.scale))),
                          (150, 150, 150), -1)
        b = self.world.bounds
        cv2.rectangle(img, self._px(b[0], b[3]), self._px(b[1], b[2]), (60, 60, 60), 1)
        if mission.path is not None:
            pts = np.array([self._px(*p) for p in mission.path.pts], np.int32)
            cv2.polylines(img, [pts], False, (180, 112, 117), 1)
        for h in hits[::2]:
            cv2.circle(img, self._px(*h), 1, (40, 40, 220), -1)
        for m in self.world.markers:
            cv2.circle(img, self._px(*m.pos[:2]), 3, (0, 0, 0), -1)
        self.true_trail.append(self._px(*x_true[:2]))
        self.est_trail.append(self._px(*x_est[:2]))
        cv2.polylines(img, [np.array(self.true_trail, np.int32)], False, (119, 158, 27), 2)
        cv2.polylines(img, [np.array(self.est_trail, np.int32)], False, (2, 95, 217), 1)
        # 2-sigma ellipse
        vals, vecs = np.linalg.eigh(P[:2, :2])
        vals = np.maximum(vals, 1e-10)
        ang = -math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
        axes = (max(1, int(2 * math.sqrt(vals[1]) * self.scale)), max(1, int(2 * math.sqrt(vals[0]) * self.scale)))
        cv2.ellipse(img, self._px(*x_est[:2]), axes, ang, 0, 360, (2, 95, 217), 1)
        for pose, col in [(x_true, (119, 158, 27)), (x_est, (2, 95, 217))]:
            p0 = self._px(*pose[:2])
            p1 = self._px(pose[0] + 0.25 * math.cos(pose[2]), pose[1] + 0.25 * math.sin(pose[2]))
            cv2.arrowedLine(img, p0, p1, col, 2, tipLength=0.3)
        e = x_est - x_true
        e[2] = wrap(e[2])
        sig = np.sqrt(np.diag(P))
        txt = (f"t={t:5.1f}s  {cmd.mode.value:<10} err={np.hypot(e[0], e[1]) * 100:4.1f}cm "
               f"(2s={2 * np.hypot(sig[0], sig[1]) * 100:4.1f}cm) yaw={math.degrees(e[2]):+4.1f}deg")
        cv2.putText(img, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "green=truth  orange=EKF+2sigma  red=LiDAR  purple=A* path", (8, self.map_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
        if cmd.safety_stop:
            cv2.putText(img, "LIDAR SAFETY STOP", (420, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return img

    def frame(self, t, x_est, P, x_true, cam_img, meas, hits, mission, cmd):
        from src.perception.aruco_detector import ArucoLandmarkDetector
        canvas = np.zeros((self.H, self.W, 3), np.uint8)
        view = self.robot.render_view("overview", 640, 480)
        canvas[120:600, 0:640] = view
        cam = ArucoLandmarkDetector.draw(cam_img, meas) if cam_img is not None else np.zeros((480, 640, 3), np.uint8)
        canvas[0:360, 640:1280] = cv2.resize(cam, (640, 360))
        canvas[360:720, 640:1280] = self._map_panel(t, x_est, P, x_true, hits, mission, cmd)
        cv2.putText(canvas, "G82 - TurtleBot3 EKF + LQR visual docking (MuJoCo)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "robot camera", (650, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.writer.write(canvas)

    def close(self):
        self.writer.release()
