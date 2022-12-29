from __future__ import annotations  # forward type hints

import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from textwrap import wrap
from typing import List

from fractals.color import Color


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
        self.original = deepcopy(colors)

    @classmethod
    def from_element(cls, element: ET.Element) -> Palette:
        count: int = int(element.attrib["count"])
        format = element.attrib["format"]
        string = "".join([x.strip() for x in element.text.split("\n")])
        colors = [Color.from_hex(c) for c in wrap(string, 6)]
        assert len(colors) == count
        return Palette(count, colors, format)

    def rotate(self, fraction: float) -> List[Color]:
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

    def animate(self, n_rotations: int, frame: int, total_frames: int) -> None:
        frames_per_rotation = total_frames / n_rotations
        frame_in_rotation = frame % frames_per_rotation
        self.colors = self.rotate(frame_in_rotation / frames_per_rotation)

    def to_element(self) -> ET.Element:
        palette = ET.Element("palette")
        string = "".join([str(c) for c in self.colors])
        palette.text = "".join([f"\n      {x}" for x in wrap(string, 48)]) + "\n"
        palette.attrib["count"] = str(self.count)
        return palette

    def __repr__(self):
        return ET.tostring(self.to_element(), encoding="utf-8").decode()
