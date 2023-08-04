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
        self.animations: List[Animation] = []

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

    def animate(self, frame):
        for animation in self.animations:
            animation.apply(self, frame)

    def __repr__(self):
        return ET.tostring(self.to_element(), encoding="utf-8").decode()


class Animation(abc.ABC):
    def __init__(
        self,
        start_frame: int,
        animation_length: int,
        transition: callable,
        reverse: bool,
        value_from: Union[Transform, float] = 0.0,
        value_to: Union[Transform, float] = 1.0,
    ):
        self.start_frame = start_frame
        self.end_frame = animation_length + start_frame
        self.reverse = reverse
        self.transition = transition
        self.value_from = value_from
        self.value_to = value_to

    def apply(self, xform: XForm, current_frame: int):
        if self.start_frame <= current_frame < self.end_frame:
            v_from = self.value_from if not self.reverse else self.value_to
            v_to = self.value_to if not self.reverse else self.value_from
            factor: Union[Transform, float] = self.transition(
                frame=current_frame - self.start_frame,
                total_frames=self.end_frame - self.start_frame,
                value_from=v_from,
                value_to=v_to,
            )
            self.animate_xform(xform, factor)
        pass

    @abc.abstractmethod
    def animate_xform(self, xform: XForm, factor: float):
        pass


class TranslationAnimation(Animation):
    def __init__(
        self,
        target_transform: Transform,
        start_frame: int,
        animation_length: int,
        transition: callable,
        reverse: bool,
        value_from: Union[Transform, float] = 0.0,
        value_to: Union[Transform, float] = 1.0,
    ):
        super().__init__(
            start_frame=start_frame,
            animation_length=animation_length,
            transition=transition,
            reverse=reverse,
            value_from=value_from,
            value_to=value_to,
        )
        self.target_transform = target_transform

    def animate_xform(self, xform: XForm, factor) -> None:
        result = self.target_transform * factor + xform.transform * (
            1 - factor
        )
        xform.transform.coefs = result.coefs


class ScalingAnimation(Animation):
    def __init__(
        self,
        start_frame: int,
        animation_length: int,
        transition: callable,
        value_to: Union[Transform, float],
        value_from: Union[Transform, float] = 1.0,
        reverse: bool = False,
    ):
        super().__init__(
            start_frame=start_frame,
            animation_length=animation_length,
            transition=transition,
            reverse=reverse,
            value_from=value_from,
            value_to=value_to,
        )

    def animate_xform(self, xform: XForm, factor) -> None:
        xform.transform.scale(factor)


class RotationAnimation(Animation):
    def __init__(
        self,
        start_frame: int,
        animation_length: int,
        transition: callable,
        reverse: bool = False,
        value_from: Union[Transform, float] = 0.0,
        value_to: Union[Transform, float] = 1.0,
    ):
        super().__init__(
            start_frame=start_frame,
            animation_length=animation_length,
            transition=transition,
            reverse=reverse,
            value_from=value_from,
            value_to=value_to,
        )

    def animate_xform(self, xform: XForm, factor) -> None:
        xform.transform.rotate(factor * 2 * math.pi)


class AttributeAnimation(Animation):
    def __init__(
        self,
        attribute: str,
        start_frame: int,
        animation_length: int,
        transition: callable,
        value_to: Union[Transform, float],
        value_from: Union[Transform, float] = None,
        reverse: bool = False,
    ):
        super().__init__(
            start_frame=start_frame,
            animation_length=animation_length,
            transition=transition,
            reverse=reverse,
            value_from=value_from,
            value_to=value_to,
        )
        self.attribute = attribute

    def apply(self, xform: XForm, current_frame: int):
        self.value_from = float(xform.element.attrib[self.attribute])
        super().apply(xform, current_frame)

    def animate_xform(self, xform: XForm, factor) -> None:
        xform.element.attrib[self.attribute] = str(round(factor, 7))


class OrbitAnimation(Animation):
    def __init__(
        self,
        radius: float,
        start_frame: int,
        animation_length: int,
        transition: callable,
        value_to: Union[Transform, float] = 1.0,
        value_from: Union[Transform, float] = 0.0,
        reverse: bool = False,
    ):
        super().__init__(
            start_frame=start_frame,
            animation_length=animation_length,
            transition=transition,
            reverse=reverse,
            value_from=value_from,
            value_to=value_to,
        )
        self.radius = radius

    def animate_xform(self, xform: XForm, factor) -> None:
        x = np.array([self.radius, 0, 1]).T
        x = rotate_vector(x, 2 * math.pi * factor)
        x[0] -= self.radius
        xform.transform.translate(x[0], x[1])


def loop(
    cls,
    animation_length: int,
    total_frames: int,
    offset: int = 0,
    stack: bool = False,
    value_to: Union[Transform, float] = 1.0,
    value_from: Union[Transform, float] = 0.0,
    **kwargs
) -> List[Animation]:
    result = []
    value_diff = 0
    if stack:
        if value_from is None:
            logger.error("when stacking, value_from is needed.")
            sys.exit(1)
        value_diff = value_to - value_from
    for i in range(total_frames):
        if i < offset:
            continue
        j = i - offset
        loop_number = i // animation_length
        frame_in_loop = j % animation_length
        if frame_in_loop == 0:
            result.append(
                cls(
                    start_frame=loop_number * animation_length,
                    animation_length=animation_length,
                    value_from=value_from,
                    value_to=value_to,
                    **kwargs
                )
            )
            if stack:
                value_from += value_diff
                value_to += value_diff
    return result
