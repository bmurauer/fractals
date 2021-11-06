from __future__ import annotations

import copy
import datetime
import math
import os
import re
import shutil
import subprocess as sp
import tempfile
import xml.etree.ElementTree as ET
from glob import glob
from textwrap import wrap
from typing import List

import numpy as np

from .utils import logger


class Transform:
    def __init__(self, string: str):
        self.coefs = [float(x) for x in string.split(" ")]
        self.coefs = np.array(self.coefs).reshape(3, 2).T

    def rotate(self, radiants: float):
        self.coefs = self.coefs.dot(
            np.array(
                [
                    [math.cos(rad), -math.sin(rad), 0],
                    [math.sin(rad), math.cos(rad), 0],
                    [0, 0, 1],
                ]
            )
        )

    def translate(self, x: float, y: float):
        self.coefs = self.coefs.dot(
            np.array(
                [
                    [1, 0, x],
                    [0, 1, y],
                    [0, 0, 1],
                ]
            )
        )

    def scale(self, x: float, y: float = None):
        self.coefs = self.coefs.dot(
            np.array(
                [
                    [x, 0, 0],
                    [0, y or x, 0],
                    [0, 0, 1],
                ]
            )
        )

    def mutate(self):
        i = lambda: random.uniform(-0.1, 0.1)
        self.rotate(i())
        self.translate(i(), i())
        self.scale(2 * i(), 2 * i())

    def __repr__(self):
        return " ".join([str(x)[:8] for x in self.coefs.T.flatten()])


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
    ):
        self.count = count
        self.format = format
        self.colors = colors

    @classmethod
    def from_element(self, element: ET.Element):
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
        for i, color in enumerate(self.colors):
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
            lower = self.colors[bound_lower]
            upper = self.colors[bound_upper]
            new_colors.append(lower.interpolate_towards(upper, fraction_between))

        self.colors = new_colors

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
        coefs: Transform,
        color: float,
    ):
        self.element = element
        self.coefs = coefs
        self.color = color

    @classmethod
    def from_element(self, element: ET.Element):
        coefs = Transform(element.attrib["coefs"])
        color = float(element.attrib["color"])
        return XForm(element, coefs, color)

    def to_element(self) -> ET.Element:
        self.element.attrib["color"] = str(self.color)
        self.element.attrib["coefs"] = str(self.coefs)
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
            self.attrib["color_speed"] = str(random.uniform(0.0, 1.0))
        if mutate_variations:
            pass

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
        clone.append(self.palette.to_element())
        return clone

    def rotate_colors(self, n_frames: int) -> Flames:
        result: List[Flame] = []
        for i in range(n_frames):
            clone = copy.deepcopy(self)
            clone.element.attrib["time"] = str(i + 1)
            clone.palette.rotate(i / float(n_frames))
            result.append(clone)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        return Flames(result, f'color-{self.element.attrib["name"]}-{timestamp}')

    def mutate(self, mutate_palette=True, mutate_xforms=True):
        if mutate_palette:
            self.palette.mutate()
        if mutate_xforms:
            [xform.mutate() for xform in self.xforms]

    def __repr__(self):
        return ET.tostring(self.to_element()).decode()


class Flames:
    def __init__(
        self,
        flames: List[Flame],
        directory: str,
        quality: int = 1000,
        supersample: int = 2,
    ):
        self.flames = flames
        self.directory = directory
        self.filename = os.path.join(self.directory, "animation.flame")
        self.moviename = os.path.join(self.directory, "animation.mp4")
        self.quality = quality
        self.supersample = supersample

    def write_file(self):
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        root = ET.Element("flames")
        [root.append(f.to_element()) for f in self.flames]
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

        logger.debug("rendering flames of %s", self.filename)
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
