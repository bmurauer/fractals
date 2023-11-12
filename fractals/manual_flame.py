import abc
import math
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import sdl2
import sdl2.ext


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __mul__(self, factor: float):
        return Point(self.x * factor, self.y * factor)

    def __repr__(self):
        return f"P[{self.x},{self.y}]"


class Variation(abc.ABC):
    def __init__(
        self,
        weight: float,
        a: float,
        b: float,
        c: float,
        d: float,
        e: float,
        f: float,
    ):
        self.weight = weight
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    @abc.abstractmethod
    def apply(self, point: Point) -> Point:
        pass


class Linear(Variation):
    def apply(self, point: Point):
        return Point(
            self.a * point.x + self.b * point.y + self.c,
            self.d * point.x + self.e * point.y + self.f,
        )


class Spherical(Variation):
    def apply(self, point: Point):
        r = math.sqrt(point.x * point.x + point.y * point.y)
        return Point(
            self.a * point.x + self.b * point.y + self.c,
            self.d * point.x + self.e * point.y + self.f,
        ) * (1 / (r * r))


class Flame:
    def __init__(self, p: float, variations: List[Variation]):
        self.p = p
        self.variations = variations

    def render(self, original_point: Point):
        point = Point(0.0, 0.0)
        for variation in self.variations:
            point += variation.apply(original_point) * variation.weight
        return point


def get_palette():
    return [i | i << 8 | i << 16 for i in range(256)]


def draw(
    flames: List[Flame],
    pixels: np.ndarray,
    dim_x: int = 300,
    dim_y: int = 300,
    samples: int = 100,
    iterations: int = 50,
    gamma: float = 2.0,
):
    assert gamma >= 1.0

    freqs = np.zeros((dim_x, dim_y))
    colors = np.zeros((dim_x, dim_y))

    for _ in range(samples):
        color = random.uniform(a=0.0, b=1.0)
        point = Point(
            random.uniform(a=-1.0, b=1.0), random.uniform(a=-1.0, b=1.0)
        )
        for i in range(iterations):
            flame = random.choices(
                flames, weights=[flame.p for flame in flames], k=1
            )[0]
            point = flame.render(point)
            if i > 20:
                x = round(point.x * (dim_x / 2))
                y = round(point.y * (dim_y / 2))
                freqs[x][y] += 1
                colors[x][y] = (colors[x][y] + color) / 2

    max_freq = np.max(freqs)
    log_max_freq = math.log(max_freq)
    max_color = np.max(colors)
    inv_gamma = 1.0 / gamma
    palette = get_palette()

    raws = np.zeros((dim_x, dim_y))
    for x in range(dim_x):
        for y in range(dim_y):
            if freqs[x][y] == 0:
                continue
            freq = freqs[x][y] / max_freq
            color = colors[x][y] / max_color
            if color == 0 or math.log(color) == 0:
                continue
            alpha = math.log(freq) / log_max_freq
            corrected = color * alpha**inv_gamma
            raws[x][y] = corrected

    max_corrected = np.max(raws)
    for x in range(dim_x):
        for y in range(dim_y):
            as_int = int(255 * raws[x][y] / max_corrected)
            pixels[x][y] = palette[as_int]


dim_x = 300
dim_y = 300
sierpinsky_flames = [
    Flame(p=1.0, variations=[Spherical(0.2, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0)]),
    Flame(p=1.0, variations=[Linear(1.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0)]),
    Flame(p=1.0, variations=[Linear(1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5)]),
    # Flame(p=1.0, variations=[Linear(1.0, 0.5, 0.0, 0.5, 0.0, 0.5, -0.5)]),
]

from sdl2 import *
import sys

SDL_Init(SDL_INIT_VIDEO)
win = SDL_CreateWindow(
    b"test", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, dim_x, dim_y, 0
)
surf = SDL_GetWindowSurface(win)

# clear surface
SDL_FillRect(surf, None, 0)

pixels = sdl2.ext.pixels2d(surf)
draw(sierpinsky_flames, pixels, iterations=400, dim_x=dim_x, dim_y=dim_y)

# signal SDL that surface is ready to be presented on screen
SDL_UpdateWindowSurface(win, surf)

# main loop
while True:
    # process events, exit if window is closed
    # (spinning event loop is a must, even if you don't react to events)
    ev = SDL_Event()
    while SDL_PollEvent(ev):
        if ev.type == SDL_QUIT:
            sys.exit(0)
