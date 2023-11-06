import abc
import math
import sys
import xml.etree.ElementTree as ET
from typing import List, Union
from copy import deepcopy

import numpy as np
from fractals.transform import Transform
from fractals.utils import rotate_vector

from logzero import logger


class XForm:
    def __init__(
        self,
        element: ET.Element,
        transform: Transform,
        color: float = 1.0,
        weight: float = 1.0,
    ):
        self.element = element
        self.transform = transform
        self.color = color
        self.weight = weight

    @classmethod
    def from_element(cls, element: ET.Element):
        transform = Transform(element.attrib["coefs"])
        color = float(element.attrib["color"])
        # final xforms don't have weight
        if "weight" in element.attrib:
            weight = float(element.attrib["weight"])
        else:
            weight = 1.0
        return XForm(element, transform, color, weight)

    def to_element(self) -> ET.Element:
        self.element.attrib["color"] = str(self.color)
        self.element.attrib["coefs"] = str(self.transform)
        self.element.attrib["weight"] = str(self.weight)
        return self.element

    # def animate(self, flames: List[Flame], frame):
    #     factor = None
    #     for animation in self.animations:
    #         new_factor = animation.apply(self, frame)
    #         if factor is None:
    #             factor = new_factor
    #         elif new_factor is not None:
    #             factor *= new_factor
    #     if factor is not None:
    #         for animation in self.animations:
    #             if animation.start_frame <= frame < animation.end_frame:
    #                 animation.animate_xform(self, factor)

    def __repr__(self):
        return ET.tostring(self.to_element(), encoding="utf-8").decode()
