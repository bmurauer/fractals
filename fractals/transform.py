from __future__ import annotations  # forward type hints
import math
import random
from typing import Union

import numpy as np
from copy import deepcopy

from fractals.utils import rotate_vector


class Transform:
    def __init__(self, string: str):
        """

        :param string:
            represented by six floats: [x0 y0 x1 y1 xb yb]
            1 and 2 define the coordinates of the first triangle corner
            3 and 4 define the coordinates of the second triangle corner
            5 and 6 define the base of the triangle.

        """
        self.coefs = [float(x) for x in string.split(" ")]
        self.coefs = np.array(self.coefs).reshape(3, 2).T
        self.coefs = np.c_[self.coefs.T, np.ones(3)].T

    @property
    def origin(self):
        return np.array([self.coefs[0][2], self.coefs[1][2]])

    def rotate(self, radians: float) -> None:
        origin_x = self.coefs[0][2]
        origin_y = self.coefs[1][2]

        self.coefs = np.array(
            [
                [math.cos(radians), -math.sin(radians), 0],
                [math.sin(radians), math.cos(radians), 0],
                [0, 0, 1],
            ]
        ).dot(self.coefs)
        self.coefs[0][2] = origin_x
        self.coefs[1][2] = origin_y

    def translate_orbit_step(
        self, radius: float, frame: int, total_frames: int
    ) -> None:
        x = np.array([radius, 0, 1]).T
        x = rotate_vector(x, 2 * math.pi * frame / total_frames)
        self.translate(x[0] - radius, x[1])

    def translate(self, x: float, y: float) -> None:
        self.coefs[0][2] += x
        self.coefs[1][2] += y

    def scale(self, x: float, y: float = None) -> None:
        self.coefs = np.array(
            [
                [x, 0, 0],
                [0, y or x, 0],
                [0, 0, 1],
            ]
        ).dot(self.coefs)

    def mutate(self) -> None:
        def i():
            return random.uniform(-0.1, 0.1)

        self.rotate(i())
        self.translate(i(), i())
        self.scale(2 * i(), 2 * i())

    def __add__(self, other: Transform) -> Transform:
        t = deepcopy(self)
        t.coefs = self.coefs + other.coefs
        t.coefs[2] = np.array([1, 1, 1])
        return t

    def __sub__(self, other) -> Transform:
        t = deepcopy(self)
        t.coefs = self.coefs - other.coefs
        t.coefs[2] = np.array([1, 1, 1])
        return t

    def __truediv__(self, other: Union[float, int]) -> Transform:
        t = deepcopy(self)
        t.coefs = self.coefs / other
        t.coefs[2] = np.array([1, 1, 1])
        return t

    def __mul__(self, other: Union[float, int]) -> Transform:
        t = deepcopy(self)
        t.coefs = self.coefs * other
        t.coefs[2] = np.array([1, 1, 1])
        return t

    def __repr__(self):
        return " ".join(
            [str(round(x, 6)) for x in self.coefs[0:2, :].T.flatten()]
        )
