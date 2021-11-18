from __future__ import annotations

import copy
import datetime
import math
import os
import random
import re
import shutil
import subprocess as sp
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from glob import glob
from textwrap import wrap
from typing import List, Union

import numpy as np

from .utils import logger


def interpolate_linear(
    anim_value: AnimationValue,
    frame: int,
    n_frames: int,
):
    frames_per_repetition = n_frames / anim_value.n_repetitions
    frame_in_repetition = frame % frames_per_repetition

    if (frame % frames_per_repetition) <= frames_per_repetition / 2:
        return (
            anim_value.original_value
            + anim_value.offset * frame_in_repetition * 2 / frames_per_repetition
        )
    else:
        return (
            anim_value.original_value
            + anim_value.offset * 2
            - anim_value.offset * frame_in_repetition * 2 / frames_per_repetition
        )


def interpolate_sinusoidal(
    anim_value: AnimationValue,
    frame: int,
    n_frames: int,
):
    # we need a function that behaves like a sinus to smoothly transition from 0 to 1
    def smooth(x):
        return (math.cos(math.pi * (2 * x + 1)) + 1) / 2

    frames_per_repetition = n_frames / anim_value.n_repetitions
    frame_in_repetition = frame % frames_per_repetition
    # no check necessary, the cosine function automagically has a bump back to 0.
    return anim_value.value + anim_value.offset * smooth(
        frame_in_repetition / frames_per_repetition
    )


def rotate_linear(
    anim_value: AnimationValue,
    frame: int,
    n_frames: int,
):
    anim_value.value.rotate(frame * anim_value.n_repetitions * 2 * math.pi / n_frames)
    return anim_value.value


def orbit_transform(
    anim_value: AnimationValue,
    frame: int,
    n_frames: int,
):
    if type(anim_value.value) is not Transform:
        raise Exception("orbit only allowed for Transforms")
    transform: Transform = anim_value.value

    # rotation root is south of the origin
    rotation_root = np.array([0, -anim_value.radius, 1]).T
    radiants = 2 * math.pi * anim_value.n_repetitions * frame / n_frames
    offset_coordinates = rotation_root.dot(
        np.array(
            [
                [math.cos(radiants), -math.sin(radiants), 0],
                [math.sin(radiants), math.cos(radiants), 0],
                [0, 0, 1],
            ]
        )
    )
    off_x, off_y, _ = offset_coordinates.tolist()

    transform.translate(off_x, off_y)
    return transform


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")


class Transform:
    def __init__(self, string: str):
        self.coefs = [float(x) for x in string.split(" ")]
        self.coefs = np.array(self.coefs).reshape(3, 2).T
        self.coefs = np.c_[self.coefs.T, np.ones(3)].T

    def rotate(self, radiants: float):
        origin_x = self.coefs[0][2]
        origin_y = self.coefs[1][2]

        self.coefs = np.array(
            [
                [math.cos(radiants), -math.sin(radiants), 0],
                [math.sin(radiants), math.cos(radiants), 0],
                [0, 0, 1],
            ]
        ).dot(self.coefs)
        self.coefs[0][2] = origin_x
        self.coefs[1][2] = origin_y

    def translate(self, x: float, y: float):
        self.coefs[0][2] += x
        self.coefs[1][2] += y

    def scale(self, x: float, y: float = None):
        self.coefs = np.array(
            [
                [x, 0, 0],
                [0, y or x, 0],
                [0, 0, 1],
            ]
        ).dot(self.coefs)

    def mutate(self):
        def i():
            return random.uniform(-0.1, 0.1)

        self.rotate(i())
        self.translate(i(), i())
        self.scale(2 * i(), 2 * i())

    def __add__(self, other: Transform):
        t = deepcopy(self)
        t.coefs = self.coefs + other.coefs
        return t

    def __sub__(self, other):
        t = deepcopy(self)
        t.coefs = self.coefs - other.coefs
        return t

    def __truediv__(self, other: Union[float, int]):
        t = deepcopy(self)
        t.coefs = self.coefs / other
        return t

    def __mul__(self, other: Union[float, int]):
        t = deepcopy(self)
        t.coefs = self.coefs * other
        return t

    def __repr__(self):
        return " ".join([str(round(x, 6)) for x in self.coefs[0:2, :].T.flatten()])


class Color:
    def __init__(self, r: int, g: int, b: int):
        self.r = r
        self.g = g
        self.b = b

    @classmethod
    def from_hex(cls, hex_string):
        red, green, blue = wrap(hex_string, 2)
        return Color(int(red, 16), int(green, 16), int(blue, 16))

    def interpolate_towards(self, other: Color, fraction: float) -> Color:
        assert 0.0 <= fraction <= 1.0
        r = round(self.r + (other.r - self.r) * fraction)
        g = round(self.g + (other.g - self.g) * fraction)
        b = round(self.b + (other.b - self.b) * fraction)
        return Color(r, g, b)

    def __repr__(self):
        return f"{self.r:02x}{self.g:02x}{self.b:02x}"


class Palette:
    def __init__(
        self,
        count: int,
        colors: List[Color],
        format: str = "rgb",
        n_rotations: int = 0,
    ):
        self.count = count
        self.format = format
        self.colors = colors
        self.original = deepcopy(colors)
        self.n_rotations = n_rotations

    @classmethod
    def from_element(cls, element: ET.Element):
        count: int = int(element.attrib["count"])
        format = element.attrib["format"]
        string = "".join([x.strip() for x in element.text.split("\n")])
        colors = [Color.from_hex(c) for c in wrap(string, 6)]
        assert len(colors) == count
        return Palette(count, colors, format)

    def rotate(self, fraction: float):

        new_colors: List[Color] = []
        # the fraction is a fraction of a full rotation.
        # measured in steps:
        fraction_in_steps = fraction * self.count

        # calculate every step
        for i, color in enumerate(self.original):
            j = i + fraction_in_steps
            # find the two defined colors that border j
            bound_lower = math.floor(j)
            bound_upper = math.ceil(j)
            if bound_lower == bound_upper:
                bound_upper += 1
            if bound_lower >= self.count:
                bound_lower -= self.count
            if bound_upper >= self.count:
                bound_upper -= self.count
            if j >= self.count:
                j -= self.count
            fraction_between = j - bound_lower
            lower = self.original[bound_lower]
            upper = self.original[bound_upper]
            new_colors.append(lower.interpolate_towards(upper, fraction_between))

        return new_colors

    def animate(self, frame, total_frames):
        frames_per_rotation = total_frames / self.n_rotations
        frame_in_rotation = frame % frames_per_rotation
        self.colors = self.rotate(frame_in_rotation / frames_per_rotation)

    def mutate(self):
        pass

    def to_element(self):
        palette = ET.Element("palette")
        string = "".join([str(c) for c in self.colors])
        palette.text = "".join([f"\n      {x}" for x in wrap(string, 48)]) + "\n"
        palette.attrib["count"] = str(self.count)
        return palette

    def __repr__(self):
        return ET.tostring(self.to_element(), encoding="utf-8").decode()


class XForm:
    def __init__(
        self,
        element: ET.Element,
        coefs: AnimationValue,
        color: AnimationValue,
        weight: AnimationValue = None,
    ):
        self.element = element
        self.coefs = coefs
        self.color = color
        self.weight = weight or AnimationValue(1.0)

    @classmethod
    def from_element(cls, element: ET.Element):
        coefs = AnimationValue(Transform(element.attrib["coefs"]))
        color = AnimationValue(float(element.attrib["color"]))
        weight = AnimationValue(float(element.attrib["weight"]))
        return XForm(element, coefs, color, weight)

    def to_element(self) -> ET.Element:
        self.element.attrib["color"] = str(self.color)
        self.element.attrib["coefs"] = str(self.coefs)
        self.element.attrib["weight"] = str(self.weight)
        return self.element

    def mutate(
        self,
        mutate_coefs=True,
        mutate_color=True,
        mutate_variations=True,
    ):
        if mutate_coefs:
            self.coefs.mutate()
        if mutate_color:
            self.color = random.uniform(0.0, 1.0)
            self.element.attrib["color_speed"] = str(random.uniform(0.0, 1.0))
        if mutate_variations:
            pass

    def animate(self, frame, total_frames):
        self.weight = self.weight.animate(frame, total_frames)
        self.coefs = self.coefs.animate(frame, total_frames)
        self.color = self.color.animate(frame, total_frames)

    def __repr__(self):
        return ET.tostring(self.to_element(), encoding="utf-8").decode()


class Flame:
    def __init__(
        self,
        element: ET.Element,
        palette: Palette,
        xforms: List[XForm],
    ):
        self.element: ET.Element = element
        self.palette: Palette = palette
        self.xforms: List[XForm] = xforms

    @classmethod
    def from_element(cls, element: ET.Element) -> Flame:
        xforms = [XForm.from_element(xform) for xform in element.findall("xform")]
        return Flame(element, Palette.from_element(element.find("palette")), xforms)

    def to_element(self) -> ET.Element:
        clone = copy.deepcopy(self.element)
        clone[:] = []
        [clone.append(xform.to_element()) for xform in self.xforms]
        [clone.extend(self.element.findall("finalxform"))]
        clone.append(self.palette.to_element())
        return clone

    def mutate(self, mutate_palette=True, mutate_xforms=True):
        if mutate_palette:
            self.palette.mutate()
        if mutate_xforms:
            [xform.mutate() for xform in self.xforms]

    def animate(self, n_frames):
        result: List[Flame] = []
        for i in range(n_frames):
            clone = copy.deepcopy(self)
            clone.element.attrib["time"] = str(i + 1)
            clone.palette.animate(i, n_frames)
            for xform in clone.xforms:
                xform.animate(i, n_frames)
            result.append(clone)
        return Flames(result, f'color-{self.element.attrib["name"]}-{get_time()}',
                      movie_file_name=self.element.attrib["name"] + '.mp4')

    def __repr__(self):
        return ET.tostring(self.to_element()).decode()


class Flames:
    def __init__(
        self,
        flames: List[Flame],
        directory: str,
        quality: int = 1000,
        supersample: int = 2,
        movie_file_name: str = "animation.mp4"
    ):
        self.flames = flames
        self.directory = directory
        self.filename = os.path.join(self.directory, "animation.flame")
        self.moviename = os.path.join(self.directory, movie_file_name)
        self.quality = quality
        self.supersample = supersample

    def write_file(self):
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        root = ET.Element("flames")
        [root.append(f.to_element()) for f in self.flames]
        logger.info("writing animation file %s", self.filename)
        ET.ElementTree(root).write(self.filename)

    def get_last_rendered_flame(self) -> int:
        pngs = sorted(glob(self.directory + "/*.png"))
        if not pngs:
            return 0
        last_png_name = os.path.basename(pngs[-1])
        pattern = r"(?P<flame_id>\d*).png"
        match = re.match(pattern, last_png_name)
        if match:
            last_flame_id = match["flame_id"]
            # check if last_flame_id is actually in the flame file
            for idx, f in enumerate(self.flames):
                if f.element.attrib["name"] == last_flame_id:
                    return idx + 1  # the last found flame should not be rendered again
        return 0

    def render(self):
        self.write_file()
        begin = self.get_last_rendered_flame()
        if begin == len(self.flames) - 1:
            logger.debug("skipping rendering, all pngs are there")
            return

        command = [
            "emberanimate",
            "--opencl",
            "--in",
            self.filename,
            "--begin",
            "begin",
            "--quality",
            "self.quality",
            "--supersample",
            "self.supersample",
        ]

        logger.info("rendering flames of %s", self.filename)
        logger.debug("command used for rendering: \n\n%s\n", " ".join(command))
        sp.Popen(command).communicate()

    def convert_to_movie(self):
        # def combine_pngs_to_mp4(output_filename: str, pngs: List[str]):
        tmpdir = tempfile.TemporaryDirectory()
        pngs = sorted(glob(self.directory + "/*.png"))
        for i, png in enumerate(pngs):
            shutil.copy(png, os.path.join(tmpdir.name, f"{i:04d}.png"))
        command = [
            "ffmpeg",
            "-i",
            f'{os.path.join(tmpdir.name, r"%04d.png")}',
            "-r",
            "25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "25",
            self.moviename,
        ]
        logger.info("combining pngs to mp4 file: %s", self.moviename)
        sp.Popen(command).communicate()
        shutil.rmtree(tmpdir.name)


class AnimationValue:
    def __init__(
        self,
        value: Union[float, Transform],
        offset: Union[float, Transform] = None,
        n_repetitions: int = 1,
        orbit: float = None,
    ):
        self.value = value
        self.original_value = deepcopy(value)
        self.method = None
        self.offset = offset
        self.n_repetitions = n_repetitions
        self.orbit = orbit

    def interpolate_linear(self, target: Union[float, Transform], n_rotations: int = 1):
        assert type(self.original_value) == type(target)
        self.offset = target
        self.n_repetitions = n_rotations
        self.method = interpolate_linear
        return self

    def interpolate_sinusoidal(
        self, target: Union[float, Transform], n_rotations: int = 1
    ):
        assert type(self.original_value) == type(target)
        self.offset = target
        self.n_repetitions = n_rotations
        self.method = interpolate_sinusoidal
        return self

    def fully_rotate_linear(self, n_rotations: int = 1):
        self.n_repetitions = n_rotations
        self.method = rotate_linear
        return self

    def orbit_transform(self, n_rotations: int = 1, radius: float = 0.1):
        self.n_repetitions = n_rotations
        self.radius = radius
        self.method = orbit_transform
        return self

    def animate(self, frame, total_frames) -> AnimationValue:
        if self.method is not None:
            self.value = self.method(self, frame, total_frames)
        return deepcopy(self)

    def reset(self):
        self.value = self.original_value

    def mutate(self):
        pass

    def __repr__(self):
        return str(self.value)
