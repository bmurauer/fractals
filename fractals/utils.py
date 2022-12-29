import logging
import math

import numpy as np

logger = logging.getLogger("Logger")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y%m%d-%H:%M:%S"
)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)


def ramp_sinusoidal(frame: int, total_frames: int) -> float:
    return (math.cos(math.pi * ((frame / total_frames) + 1)) + 1) / 2


def ramp_linear(frame: int, total_frames: int) -> float:
    return frame / total_frames


def ramp_sigmoid(frame: int, total_frames: int, bumpyness: float) -> float:
    linear = ramp_linear(frame, total_frames)
    sigmoid = 2 / (
        1 + math.pow(math.e, -((10 / total_frames) * (frame - total_frames)))
    )
    return sigmoid * bumpyness + linear * (1 - bumpyness)


def ramp_inverse_sigmoid(frame: int, total_frames: int, bumpyness: float) -> float:
    linear = ramp_linear(frame, total_frames)
    inv_sigmoid = 2 / (1 + math.pow(math.e, -((10 / total_frames) * (frame)))) - 1
    return inv_sigmoid * bumpyness + linear * (1 - bumpyness)


def rotate_vector(
    vector: np.ndarray, rad: float, rotation_center: np.ndarray = None
) -> np.ndarray:
    if rotation_center is None:
        rotation_center = np.array([0, 0, 1]).T
    x = np.array(
        [
            [math.cos(rad), -math.sin(rad), 0],
            [math.sin(rad), math.cos(rad), 0],
            [0, 0, 1],
        ]
    ).dot(vector - rotation_center)
    return x + rotation_center
