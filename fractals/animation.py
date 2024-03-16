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
        xform_index: int,
        start_frame: int,
        end_frame: int,
        method: FunctionForm = FunctionForm.INVERSE_SIGMOID,
        value_from: Optional[Union[Transform, float]] = None,
        value_to: Optional[Union[Transform, float]] = None,
    ):
        self.xform_index = xform_index
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.method: FunctionForm = method
        self.value_from = value_from
        self.value_to = value_to

    @abc.abstractmethod
    def get_value(self, xform: XForm) -> Union[Transform, float]:
        pass

    def apply(self, frame, flame) -> None:
        if frame < self.start_frame or frame > self.end_frame:
            return

        if self.value_from is None:
            first_xform = flame.xforms[self.xform_index]
            self.value_from = self.get_value(first_xform)
            logger.info(
                "set missing value_from to first xframe, =%s",
                str(self.value_from),
            )

        if self.value_to is None:
            last_xform = flame.xforms[self.xform_index]
            self.value_to = self.get_value(last_xform)
            logger.info(
                "set missing value_to to last xframe, =%s", str(self.value_to)
            )

        assert len(flame.xforms) > self.xform_index
        factor: Union[Transform, float] = transition(
            method=self.method,
            frame=frame - self.start_frame,
            total_frames=self.end_frame - self.start_frame,
            value_from=0,
            value_to=1,
        )
        new_value = (
            self.value_from + (self.value_to - self.value_from) * factor
        )

        logger.debug(
            "frame=%d, factor=%f, value_from=%s, value_to=%s, " "new_value=%s",
            frame,
            factor,
            str(self.value_from),
            str(self.value_to),
            str(new_value),
        )
        xform = flame.xforms[self.xform_index]
        self.animate_xform(xform, factor, new_value)

    @abc.abstractmethod
    def animate_xform(
        self, xform: XForm, factor: float, new_value: Union[float, Transform]
    ):
        pass


class TranslationAnimation(Animation):
    def get_value(self, xform: XForm) -> Union[Transform, float]:
        return xform.transform

    def animate_xform(
        self, xform: XForm, factor: float, new_value: Union[float, Transform]
    ) -> None:
        xform.transform.coefs = new_value.coefs


class ScalingAnimation(Animation):
    def get_value(self, xform: XForm) -> Union[Transform, float]:
        return xform.transform

    def animate_xform(
        self, xform: XForm, factor, new_value: Union[float, Transform]
    ) -> None:
        xform.transform.scale(new_value)


class RotationAnimation(Animation):
    def get_value(self, xform: XForm) -> Union[Transform, float]:
        return xform.transform

    def animate_xform(
        self, xform: XForm, factor, new_value: Union[float, Transform]
    ) -> None:
        xform.transform.rotate(new_value * 2 * math.pi)


class AttributeAnimation(Animation):
    def __init__(self, attribute: str, **kwargs):
        super().__init__(**kwargs)
        self.attribute = attribute

    def get_value(self, xform: XForm) -> Union[Transform, float]:
        return float(xform.element.attrib[self.attribute])

    def animate_xform(
        self, xform: XForm, factor: float, new_value: Union[float, Transform]
    ) -> None:
        xform.element.attrib[self.attribute] = str(round(new_value, 7))


class OrbitAnimation(Animation):
    def __init__(self, radius: float, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius

    def get_value(self, xform: XForm) -> Union[Transform, float]:
        return xform.transform

    def animate_xform(
        self, xform: XForm, factor, new_value: Union[float, Transform]
    ) -> None:
        x = np.array([self.radius, 0, 1]).T
        x = rotate_vector(x, 2 * math.pi * factor)
        x[0] -= self.radius
        xform.transform.translate(float(x[0]), float(x[1]))


def loop(
    cls,
    animation_length: int,
    total_frames: int,
    offset: int = 0,
    stack: bool = False,
    value_to: Union[Transform, float] = 1.0,
    value_from: Union[Transform, float] = 0.0,
    **kwargs,
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
                    **kwargs,
                )
            )
            if stack:
                value_from += value_diff
                value_to += value_diff
    return result
