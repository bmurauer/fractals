import abc
import math
import sys
from typing import Union, List, Optional

import numpy as np
from logzero import logger

from fractals.flame import Flame
from fractals.transform import Transform
from fractals.transitions import FunctionForm, transition
from fractals.utils import rotate_vector
from fractals.xform import XForm


class FlameAnimation:
    def __init__(
        self,
        start_frame: int,
        end_frame: int,
        method: FunctionForm = FunctionForm.INVERSE_SIGMOID,
        value_from: Optional[Union[Transform, float]] = None,
        value_to: Optional[Union[Transform, float]] = None,
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
        flame: Flame,
        value: Union[float, Transform],
    ) -> None:
        pass


class AttributeFlameAnimation(FlameAnimation):
    def __init__(self, attribute: str, **kwargs):
        super().__init__(**kwargs)
        self.attribute = attribute

    def animate_flame(
        self,
        flame: Flame,
        value: Union[float, Transform],
    ) -> None:
        flame.element.attrib[self.attribute] = str(round(value, 7))


class PaletteRotationFlameAnimation(FlameAnimation):
    def animate_flame(
        self,
        flame: Flame,
        value: Union[float, Transform],
    ) -> None:
        flame.palette.rotate(value)
