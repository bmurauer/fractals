import abc
import math
import sys
from typing import Union, List, Optional

import numpy as np
from logzero import logger

from fractals.transform import Transform
from fractals.transitions import FunctionForm, transition
from fractals.utils import rotate_vector
from fractals.xform import XForm


class Animation(abc.ABC):
    def __init__(
        self,
        start_frame: int,
        end_frame: int,
        value_from: Optional[Union[Transform, float]] = None,
        value_to: Optional[Union[Transform, float]] = None,
        method: FunctionForm = FunctionForm.INVERSE_SIGMOID,
        debug: bool = False,
    ):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.method: FunctionForm = method
        self.value_from = value_from
        self.value_to = value_to
        self.debug = debug
        self.factors = []
        self.values = []
        self.value_diffs = []

        for i, frame in enumerate(range(self.start_frame, self.end_frame)):
            previous_value = value_from
            if i > 0:
                previous_value = self.values[i - 1]
            factor = transition(
                method=self.method,
                frame=frame - self.start_frame,
                total_frames=self.end_frame - self.start_frame,
            )
            new_value = (
                self.value_from + (self.value_to - self.value_from) * factor
            )
            self.factors.append(factor)
            self.values.append(new_value)
            self.value_diffs.append(new_value - previous_value)


class XformAnimation(Animation):
    def __init__(self, xform_index: int, **kwargs):
        super().__init__(**kwargs)
        self.xform_index = xform_index

    def apply(self, frame: int, flame) -> None:
        if frame < self.start_frame or self.end_frame <= frame:
            return

        xform = flame.xforms[self.xform_index]
        value = self.values[frame - self.start_frame]
        value_diff = self.value_diffs[frame - self.start_frame]

        if self.debug:
            logger.debug(
                "frame=%d, value_from=%s, value_to=%s, value=%s, δ=%s",
                frame,
                str(self.value_from),
                str(self.value_to),
                str(value),
                str(value_diff),
            )

        self.animate_xform(xform=xform, value=value, value_diff=value_diff)

    @abc.abstractmethod
    def animate_xform(
        self,
        xform: XForm,
        value: Union[float, Transform],
        value_diff: Union[float, Transform],
    ):
        pass


class TranslationXformAnimation(XformAnimation):
    def get_value(self, xform: XForm) -> Union[Transform, float]:
        return xform.transform

    def animate_xform(
        self,
        xform: XForm,
        value: Union[float, Transform],
        value_diff: Union[float, Transform],
    ) -> None:
        xform.transform.coefs = value.coefs


class ScalingXformAnimation(XformAnimation):
    def animate_xform(
        self,
        xform: XForm,
        value: Union[float, Transform],
        value_diff: Union[float, Transform],
    ) -> None:
        xform.transform.scale(1.0 + value_diff)


class RotationXformAnimation(XformAnimation):
    def animate_xform(
        self,
        xform: XForm,
        value: Union[float, Transform],
        value_diff: Union[float, Transform],
    ) -> None:
        xform.transform.rotate(value_diff * 2 * math.pi)


class AttributeXformAnimation(XformAnimation):
    def __init__(self, attribute: str, **kwargs):
        super().__init__(**kwargs)
        self.attribute = attribute

    def animate_xform(
        self,
        xform: XForm,
        value: Union[float, Transform],
        value_diff: Union[float, Transform],
    ) -> None:
        xform.element.attrib[self.attribute] = str(round(value, 7))


class OrbitXformAnimation(XformAnimation):
    def __init__(self, radius: float, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius
        self.rotation_center = np.array([self.radius, 0, 1])
        self.rotated_vector = np.array([0, 0, 1]).T

    def animate_xform(
        self,
        xform: XForm,
        value: Union[float, Transform],
        value_diff: Union[float, Transform],
    ) -> None:
        after = rotate_vector(
            self.rotated_vector, 2 * math.pi * value_diff, self.rotation_center
        )
        diff = after - self.rotated_vector

        xform.transform.translate(float(diff[0]), float(diff[1]))
        self.rotated_vector = after


class FlameAnimation(Animation):
    def apply(self, frame: int, flame) -> None:
        if frame < self.start_frame or self.end_frame <= frame:
            return

        value = self.values[frame - self.start_frame]
        value_diff = self.value_diffs[frame - self.start_frame]

        if self.debug:
            logger.debug(
                "frame=%d, value_from=%s, value_to=%s, value=%s, δ=%s",
                frame,
                str(self.value_from),
                str(self.value_to),
                str(value),
                str(value_diff),
            )

        self.animate_flame(flame=flame, value=value)

    @abc.abstractmethod
    def animate_flame(
        self,
        flame,
        value: Union[float, Transform],
    ) -> None:
        pass


class AttributeFlameAnimation(FlameAnimation):
    def __init__(self, attribute: str, **kwargs):
        super().__init__(**kwargs)
        self.attribute = attribute

    def animate_flame(
        self,
        flame,
        value: Union[float, Transform],
    ) -> None:
        flame.element.attrib[self.attribute] = str(round(value, 7))


class PaletteRotationFlameAnimation(FlameAnimation):
    def animate_flame(
        self,
        flame,
        value: Union[float, Transform],
    ) -> None:
        flame.palette.rotate(value)


def bounce(
    cls,
    start_frame: int,
    end_frame: int,
    value_from: Union[Transform, float],
    value_to: Union[Transform, float],
    **kwargs,
) -> List[Animation]:
    half = (end_frame - start_frame) // 2
    return [
        cls(
            start_frame=start_frame,
            end_frame=start_frame + half,
            value_from=value_from,
            value_to=value_to,
            **kwargs,
        ),
        cls(
            start_frame=start_frame + half + 1,
            end_frame=end_frame,
            value_from=value_to,
            value_to=value_from,
            **kwargs,
        ),
    ]


def loop(
    cls,
    animation_length: int,
    total_frames: int,
    offset: int = 0,
    value_to: Union[Transform, float] = 1.0,
    value_from: Union[Transform, float] = 0.0,
    **kwargs,
) -> List[Animation]:
    result = []
    for i in range(total_frames):
        if i < offset:
            continue
        j = i - offset
        loop_number = i // animation_length
        frame_in_loop = j % animation_length
        if frame_in_loop == 0:
            start_frame = loop_number * animation_length
            end_frame = start_frame + animation_length
            result.append(
                cls(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    value_from=value_from,
                    value_to=value_to,
                    **kwargs,
                )
            )
    return result
