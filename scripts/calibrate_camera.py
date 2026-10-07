"""Checkerboard calibration of the robot camera inside MuJoCo (criterion B).

A 9x6 (inner corners), 25 mm checkerboard is a mocap body placed at random poses
in front of the TurtleBot camera. Images are rendered through the same noise
model as the pipeline, corners are found with findChessboardCorners+cornerSubPix,
and cv2.calibrateCamera estimates K and distortion. Outputs:
  * config/camera_calibration.json  (used by the pipeline if present)
  * results/calibration/report.md   K vs analytic K from MuJoCo's fovy, reprojection RMS,
                                    and an extrinsic check of T_B_C (board centre
                                    predicted through T_B_C vs measured by PnP)
  * results/calibration/views.png
    python scripts/calibrate_camera.py [--views 30]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src  # noqa: F401,E402

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from src.config import ASSETS, RESULTS, ROOT, make_config  # noqa: E402
from src.perception.camera_model import CameraIntrinsics, T_B_C  # noqa: E402

COLS, ROWS, SQ = 9, 6, 0.025          # inner corners, square size [m]


def board_xml() -> str:
    geoms = []
    nx, ny = COLS + 1, ROWS + 1
    w, h = nx * SQ, ny * SQ
    geoms.append(f'<geom type="box" size="{w / 2 + SQ} {h / 2 + SQ} 0.002" pos="0 0 -0.0025" rgba="1 1 1 1" '
                 f'contype="0" conaffinity="0"/>')
    for i in range(nx):
        for j in range(ny):
            if (i + j) % 2 == 0:
                x = -w / 2 + (i + 0.5) * SQ
                y = -h / 2 + (j + 0.5) * SQ
                geoms.append(f'<geom type="box" size="{SQ / 2} {SQ / 2} 0.0005" pos="{x:.4f} {y:.4f} 0" '
                             f'rgba="0 0 0 1" contype="0" conaffinity="0"/>')
    return "\n".join(geoms)


def build_scene() -> Path:
    xml = f"""<mujoco><include file="turtlebot3_waffle_pi.xml"/>
<visual><headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0 0 0"/></visual>
<worldbody><light pos="0 0 3" dir="0 0 -1" directional="true"/>
<geom type="plane" size="4 4 0.1" rgba="0.55 0.55 0.55 1"/>
<body name="board" mocap="true" pos="0.6 0 0.2">{board_xml()}</body>
</worldbody></mujoco>"""
    p = ASSETS / "tb3" / "_calib_scene.xml"
    p.write_text(xml)
    return p


def object_points() -> np.ndarray:
    # inner corners in the board frame (z = 0 plane)
    w, h = (COLS + 1) * SQ, (ROWS + 1) * SQ
    pts = [(-w / 2 + (c + 1) * SQ, -h / 2 + (r + 1) * SQ, 0.0) for r in range(ROWS) for c in range(COLS)]
    return np.array(pts, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=60)
    args = ap.parse_args()
    cfg = make_config()
    rng = np.random.default_rng(7)
    path = build_scene()
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    renderer = mujoco.Renderer(m, cfg.sim.cam_height, cfg.sim.cam_width)
    opt = mujoco.MjvOption()
    opt.geomgroup[2] = 0
    cam_b = T_B_C()
    obj = object_points()
    obj_pts, img_pts, true_centres_C, views = [], [], [], []
    tries = 0
    while len(obj_pts) < args.views and tries < args.views * 4:
        tries += 1
        dist = rng.uniform(0.3, 0.7)
        lateral = rng.uniform(-0.12, 0.12) * dist / 0.5
        height = 0.107 + rng.uniform(-0.06, 0.06) * dist / 0.5
        # nominal board axes in B: x = image right (-y_B), y = up (+z_B), z = normal towards the camera (-x_B)
        R_nom = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        tilt = Rotation.from_euler("xyz", [rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-30, 30)],
                                   degrees=True).as_matrix()
        R_wb = Rotation.from_matrix(R_nom @ tilt)      # tilt about the board's own axes
        pos = np.array([0.0865 + dist, lateral, height])
        q = R_wb.as_quat()  # x y z w
        d.mocap_pos[0] = pos
        d.mocap_quat[0] = [q[3], q[0], q[1], q[2]]
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera="rgb_cam", scene_option=opt)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = np.clip(img + rng.normal(0, cfg.noise.image_noise_std, img.shape), 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, (COLS, ROWS), None)
        if not ok:
            continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
        obj_pts.append(obj)
        img_pts.append(corners)
        # true board centre in the optical frame C (robot base is at the world origin)
        p_B = np.append(pos, 1.0)
        true_centres_C.append((np.linalg.inv(cam_b) @ p_B)[:3])
        if len(views) < 6:
            v = img.copy()
            cv2.drawChessboardCorners(v, (COLS, ROWS), corners, ok)
            views.append(cv2.resize(v, (320, 240)))
    renderer.close()
    path.unlink(missing_ok=True)
    W, H = cfg.sim.cam_width, cfg.sim.cam_height
    rms, K, dist_c, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (W, H), None, None,
                                                       flags=cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
    K_an = CameraIntrinsics.analytic(cfg).K
    # extrinsic / frame check: PnP board centre (calibrated K) vs. prediction through T_B_C
    errs = []
    for rv, tv, c_true in zip(rvecs, tvecs, true_centres_C):
        R = cv2.Rodrigues(rv)[0]
        centre = (R @ obj.mean(0) + tv.ravel())
        errs.append(np.linalg.norm(centre - c_true))
    errs = np.array(errs)
    out_dir = RESULTS / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    if views:
        while len(views) % 3:
            views.append(np.zeros_like(views[0]))
        grid = np.vstack([np.hstack(views[i:i + 3]) for i in range(0, len(views), 3)])
        cv2.imwrite(str(out_dir / "views.png"), grid)
    rel = lambda a, b: 100 * (a - b) / b  # noqa: E731
    report = f"""# Camera calibration (generated by scripts/calibrate_camera.py)

Views used: {len(obj_pts)} / {tries} rendered (checkerboard {COLS}x{ROWS} inner corners, {SQ * 1000:.0f} mm squares)
Image noise model: Gaussian blur 3x3 + N(0, {cfg.noise.image_noise_std}) grey levels

| parameter | calibrated | analytic (MuJoCo fovy={cfg.sim.cam_fovy_deg} deg) | difference |
|---|---|---|---|
| fx [px] | {K[0, 0]:.2f} | {K_an[0, 0]:.2f} | {rel(K[0, 0], K_an[0, 0]):+.2f} % |
| fy [px] | {K[1, 1]:.2f} | {K_an[1, 1]:.2f} | {rel(K[1, 1], K_an[1, 1]):+.2f} % |
| cx [px] | {K[0, 2]:.2f} | {K_an[0, 2]:.2f} | {K[0, 2] - K_an[0, 2]:+.2f} px |
| cy [px] | {K[1, 2]:.2f} | {K_an[1, 2]:.2f} | {K[1, 2] - K_an[1, 2]:+.2f} px |
| k1, k2 | {dist_c.ravel()[0]:+.4f}, {dist_c.ravel()[1]:+.4f} | 0, 0 (ideal pinhole) | |

Reprojection RMS: **{rms:.3f} px**

Frame check (T_B_C): board centre measured by PnP vs. predicted from the true board pose through
T_B_C: mean {errs.mean() * 1000:.2f} mm, max {errs.max() * 1000:.2f} mm over {len(errs)} views.
A wrong axis convention in T_B_C (e.g. using MuJoCo's camera frame instead of the OpenCV optical
frame) would show up here as errors of tens of centimetres.
"""
    (out_dir / "report.md").write_text(report)
    (ROOT / "config").mkdir(exist_ok=True)
    (ROOT / "config" / "camera_calibration.json").write_text(json.dumps({
        "K": K.tolist(), "dist": dist_c.ravel().tolist(), "width": W, "height": H, "rms_px": rms,
        "n_views": len(obj_pts)}, indent=2))
    print(report)


if __name__ == "__main__":
    main()
