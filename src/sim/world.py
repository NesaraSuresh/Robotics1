"""Procedural MuJoCo world: room, random obstacles, ArUco dock + wall landmarks.

The scene is written as MJCF text (version-stable, no MjSpec dependency) next to
the TurtleBot3 model so the robot <include> and its meshes resolve.

ArUco markers are built geometrically from the dictionary bit pattern
(one thin black box per black cell on a white board). This avoids texture
orientation pitfalls and makes the marker geometry exact: the PnP object
points used by perception/aruco.py correspond 1:1 to these boxes.

World frame W: origin at room centre, x right (towards the dock), y up, z up.
Marker frame M (OpenCV ArUco convention): origin at marker centre,
x to the viewer's right, y up, z = outward normal (towards the camera).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from src.config import ASSETS, Config

ARUCO_DICT = cv2.aruco.DICT_4X4_50
ARUCO_CELLS = 6  # 4x4 data bits + 1-cell black border each side


@dataclass
class Marker:
    id: int
    pos: np.ndarray          # marker centre in W [x, y, z]
    yaw: float               # direction of the outward normal in W (rad)
    size: float              # black-border edge length [m]

    @property
    def normal(self) -> np.ndarray:
        return np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])

    def T_world_marker(self) -> np.ndarray:
        """4x4 pose of marker frame M in W."""
        n = self.normal
        z_m = n
        y_m = np.array([0.0, 0.0, 1.0])
        x_m = np.cross(y_m, z_m)  # viewer's right
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x_m, y_m, z_m, self.pos
        return T


@dataclass
class Obstacle:
    kind: str                # "box" | "cylinder"
    pos: Tuple[float, float]
    size: Tuple[float, ...]  # box: (hx, hy) ; cylinder: (radius,)
    yaw: float = 0.0


@dataclass
class WorldInfo:
    xml_path: Path
    markers: List[Marker]
    obstacles: List[Obstacle]
    dock_marker: Marker
    dock_target: np.ndarray          # (x, y, yaw) base pose when docked
    predock_target: np.ndarray       # (x, y, yaw) approach pose
    true_start: np.ndarray           # ground-truth spawn pose (evaluation only)
    nominal_start: np.ndarray        # pose the robot is told it starts at
    bounds: Tuple[float, float, float, float] = field(default=(-3, 3, -2.5, 2.5))

    def marker_map(self) -> dict:
        """Known landmark map {id: (x, y, yaw, size)} given to the estimator."""
        return {m.id: np.array([m.pos[0], m.pos[1], m.yaw, m.size]) for m in self.markers}


def _fmt(v) -> str:
    return " ".join(f"{x:.5g}" for x in v)


def _marker_xml(m: Marker, quiet_cells: float = 1.0) -> str:
    """White board + black cell boxes for one ArUco marker."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    bits = cv2.aruco.generateImageMarker(dictionary, m.id, ARUCO_CELLS)  # 6x6, 0 = black
    cell = m.size / ARUCO_CELLS
    T = m.T_world_marker()
    z_m, c = T[:3, 2], T[:3, 3]
    # board: thin box whose front face is the marker plane
    half_board = m.size / 2 + quiet_cells * cell
    t_board = 0.002
    board_c = c - z_m * t_board
    # box local axes: x along normal (thickness), y along marker x, z up
    euler = f"0 0 {m.yaw:.6f}"  # compiler angle="radian" (set by the TB3 model)
    xml = [f'<body name="marker_{m.id}" pos="{_fmt(board_c)}" euler="{euler}">',
           f'  <geom type="box" size="{t_board} {half_board:.5f} {half_board:.5f}" rgba="1 1 1 1" group="1" contype="0" conaffinity="0"/>']
    t_cell = 0.0004
    for i in range(ARUCO_CELLS):
        for j in range(ARUCO_CELLS):
            if bits[i, j] != 0:
                continue
            # in body frame: x = normal, -y = viewer's right? body y axis = R(yaw) * (0,1,0)
            # marker x_m = cross(up, n) = (-sin yaw, cos yaw, 0) = body +y.
            u = (j - (ARUCO_CELLS - 1) / 2) * cell     # along marker x (viewer right)
            v = ((ARUCO_CELLS - 1) / 2 - i) * cell     # along marker y (up)
            xml.append(f'  <geom type="box" pos="{t_board + t_cell:.5f} {u:.5f} {v:.5f}" '
                       f'size="{t_cell} {cell / 2:.5f} {cell / 2:.5f}" rgba="0 0 0 1" group="1" contype="0" conaffinity="0"/>')
    xml.append("</body>")
    return "\n".join(xml)


def _random_obstacles(cfg: Config, rng: np.random.Generator, dock_xy, start_xy) -> List[Obstacle]:
    w = cfg.world
    obs: List[Obstacle] = []
    tries = 0
    while len(obs) < w.n_obstacles and tries < 2000:
        tries += 1
        kind = "box" if rng.random() < 0.6 else "cylinder"
        if kind == "box":
            size = (rng.uniform(0.12, 0.30), rng.uniform(0.12, 0.30))
            r_eff = math.hypot(*size)
        else:
            size = (rng.uniform(0.10, 0.20),)
            r_eff = size[0]
        x = rng.uniform(-w.room_x / 2 + 0.7, w.room_x / 2 - 1.3)
        y = rng.uniform(-w.room_y / 2 + 0.7, w.room_y / 2 - 0.7)
        if math.hypot(x - start_xy[0], y - start_xy[1]) < 0.8 + r_eff:
            continue
        # keep the docking corridor clear
        if x > dock_xy[0] - 1.6 and abs(y - dock_xy[1]) < 0.75 + r_eff:
            continue
        if any(math.hypot(x - o.pos[0], y - o.pos[1]) < 0.75 + r_eff for o in obs):
            continue
        obs.append(Obstacle(kind, (x, y), size, float(rng.uniform(-math.pi, math.pi))))
    return obs


def _fixed_obstacles() -> List[Obstacle]:
    return [Obstacle("box", (-0.6, -0.6), (0.25, 0.25)), Obstacle("cylinder", (0.6, 0.9), (0.18,)),
            Obstacle("box", (1.0, -1.2), (0.2, 0.3)), Obstacle("cylinder", (-1.3, 1.2), (0.15,)),
            Obstacle("box", (-0.2, 1.8), (0.3, 0.15))]


def build_world(cfg: Config, rng: np.random.Generator, tag: str = "") -> WorldInfo:
    w = cfg.world
    hx, hy = w.room_x / 2, w.room_y / 2

    # --- markers -----------------------------------------------------------
    panel_t = 0.02
    dock_plane_x = hx - 2 * panel_t
    dock = Marker(w.dock_marker_id, np.array([dock_plane_x, w.dock_y, w.dock_marker_height]),
                  math.pi, w.dock_marker_size)
    markers = [dock]
    if w.use_landmarks:
        s, h = w.landmark_marker_size, w.landmark_marker_height
        lm = [(1, (-hx, 0.8), 0.0), (2, (-1.0, hy), -math.pi / 2), (3, (0.8, -hy), math.pi / 2),
              (4, (1.6, hy), -math.pi / 2), (5, (hx, -1.6), math.pi), (6, (-hx, -0.9), 0.0)]
        for mid, (x, y), yaw in lm:
            markers.append(Marker(mid, np.array([x, y, h]), yaw, s))

    # --- start + dock poses --------------------------------------------------
    nominal = np.array(w.start_nominal, dtype=float)
    true_start = nominal + rng.normal(0.0, w.start_sigma)
    n = dock.normal
    dock_yaw = math.atan2(-n[1], -n[0])
    dock_target = np.array([dock.pos[0] + n[0] * cfg.mission.dock_distance,
                            dock.pos[1] + n[1] * cfg.mission.dock_distance, dock_yaw])
    predock = np.array([dock.pos[0] + n[0] * cfg.mission.predock_distance,
                        dock.pos[1] + n[1] * cfg.mission.predock_distance, dock_yaw])

    obstacles = (_random_obstacles(cfg, rng, dock_target[:2], nominal[:2])
                 if w.randomise_obstacles else _fixed_obstacles())

    # --- MJCF ----------------------------------------------------------------
    wall_t = 0.05
    H = w.wall_height
    body = []
    for name, pos, size in [
        ("wall_px", (hx + wall_t, 0, H / 2), (wall_t, hy + 2 * wall_t, H / 2)),
        ("wall_nx", (-hx - wall_t, 0, H / 2), (wall_t, hy + 2 * wall_t, H / 2)),
        ("wall_py", (0, hy + wall_t, H / 2), (hx, wall_t, H / 2)),
        ("wall_ny", (0, -hy - wall_t, H / 2), (hx, wall_t, H / 2)),
    ]:
        body.append(f'<geom name="{name}" type="box" pos="{_fmt(pos)}" size="{_fmt(size)}" material="wall"/>')
    # dock: back panel + low (visual) charging pad
    body.append(f'<geom name="dock_panel" type="box" pos="{hx - panel_t:.4f} {w.dock_y} 0.2" '
                f'size="{panel_t} 0.22 0.2" rgba="0.15 0.35 0.75 1"/>')
    pad_len = cfg.mission.dock_distance - cfg.robot.front_offset - 0.015
    body.append(f'<geom name="dock_pad" type="box" pos="{dock_plane_x - pad_len / 2:.4f} {w.dock_y} 0.004" '
                f'size="{pad_len / 2:.4f} 0.15 0.004" rgba="0.95 0.75 0.1 1" contype="0" conaffinity="0"/>')
    for k, o in enumerate(obstacles):
        if o.kind == "box":
            body.append(f'<geom name="obs_{k}" type="box" pos="{o.pos[0]:.4f} {o.pos[1]:.4f} 0.2" '
                        f'size="{o.size[0]:.4f} {o.size[1]:.4f} 0.2" euler="0 0 {o.yaw:.4f}" material="obstacle"/>')
        else:
            body.append(f'<geom name="obs_{k}" type="cylinder" pos="{o.pos[0]:.4f} {o.pos[1]:.4f} 0.2" '
                        f'size="{o.size[0]:.4f} 0.2" material="obstacle2"/>')
    for m in markers:
        body.append(_marker_xml(m))

    xml = f"""<mujoco model="g82_track_a">
  <include file="turtlebot3_waffle_pi.xml"/>
  <option timestep="{cfg.sim.physics_dt}" noslip_iterations="5"/>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global offwidth="1280" offheight="960" azimuth="90" elevation="-55"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.7 0.8" rgb2="0.2 0.25 0.3" width="256" height="1536"/>
    <texture type="2d" name="floor" builtin="checker" rgb1="0.56 0.56 0.54" rgb2="0.60 0.60 0.58" width="256" height="256"/>
    <material name="floor" texture="floor" texrepeat="12 10" reflectance="0.0"/>
    <material name="wall" rgba="0.85 0.85 0.82 1"/>
    <material name="obstacle" rgba="0.85 0.45 0.2 1"/>
    <material name="obstacle2" rgba="0.3 0.6 0.4 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5" castshadow="false"/>
    <geom name="floor" type="plane" size="{hx + 1} {hy + 1} 0.1" material="floor"/>
    <camera name="overview" pos="0 -5.2 5.6" xyaxes="1 0 0 0 0.73 0.68"/>
    <camera name="top" pos="0 0 7.8" xyaxes="1 0 0 0 1 0"/>
    {chr(10).join('    ' + b for b in body)}
  </worldbody>
</mujoco>
"""
    gen_dir = ASSETS / "tb3"
    path = gen_dir / f"_scene_{os.getpid()}{tag}.xml"
    path.write_text(xml)
    return WorldInfo(path, markers, obstacles, dock, dock_target, predock, true_start, nominal,
                     (-hx, hx, -hy, hy))
