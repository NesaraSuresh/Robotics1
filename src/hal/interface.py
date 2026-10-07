"""Hardware Abstraction Layer (lecture slide: "Architecture for Sim-to-Real Readiness").

The estimator, planner and controller only ever talk to a RobotInterface.
Swapping simulation for the real TurtleBot3 means writing one new class
(e.g. a ROS 2 node wrapper) that implements these five methods; nothing in
src/estimation, src/planning or src/control changes.

Note what is *not* here: there is no method that returns the robot's true pose.
Ground truth lives in src/sim/ground_truth.py and is only used by evaluation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EncoderReading:
    t: float
    ticks_left: int
    ticks_right: int


@dataclass
class LaserScan:
    t: float
    angles: np.ndarray      # [rad] in the lidar frame
    ranges: np.ndarray      # [m], np.inf for no return


@dataclass
class CameraImage:
    t: float
    image: np.ndarray       # HxWx3 uint8, BGR (OpenCV order)


class RobotInterface(ABC):
    ticks_per_rev: int

    @abstractmethod
    def time(self) -> float: ...

    @abstractmethod
    def read_encoders(self) -> EncoderReading: ...

    @abstractmethod
    def read_scan(self) -> Optional[LaserScan]:
        """Latest scan if a new one is available this step, else None."""

    @abstractmethod
    def read_image(self) -> Optional[CameraImage]:
        """Latest image if a new one is available this step, else None."""

    @abstractmethod
    def send_velocity(self, v: float, w: float) -> None:
        """Body-frame command: forward speed v [m/s], yaw rate w [rad/s]."""

    @abstractmethod
    def step(self) -> None:
        """Advance one control period (sim: integrate physics; real: sleep)."""
