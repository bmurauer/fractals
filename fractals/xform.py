import math
import xml.etree.ElementTree as ET

import numpy as np

from fractals.transform import Transform
from fractals.transitions import beating, beating_up_down, repeat
from fractals.utils import logger, rotate_vector


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
        self.animations = dict()

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

    def animate(self, frame, total_frames):
        for animation, conf in self.animations.items():
            if animation == "attrib":
                for name, attrib_conf in conf.items():
                    old = float(self.element.attrib.get(name) or 0)
                    new = repeat(
                        frame=frame,
                        total_frames=total_frames,
                        value_from=old,
                        **attrib_conf,
                    )
                    # flame xml parsing only allows 7 decimal places
                    self.element.attrib[name] = str(round(new, 7))
            elif animation == "orbit":
                radius = conf["radius"]
                del conf["radius"]
                factor = repeat(
                    frame=frame,
                    total_frames=total_frames,
                    value_from=0,
                    value_to=1,
                    **conf,
                )
                x = np.array([radius, 0, 1]).T
                x = rotate_vector(x, 2 * math.pi * factor)
                x[0] -= radius
                self.transform.translate(x[0], x[1])
            elif animation == "translate":
                self.transform = repeat(
                    frame=frame,
                    value_from=self.transform,
                    total_frames=total_frames,
                    **conf,
                )
            elif animation == "scale":
                factor = repeat(
                    frame=frame,
                    total_frames=total_frames,
                    **conf,
                )
                self.transform.scale(factor)
            elif animation == "rotate":
                factor = repeat(
                    frame=frame,
                    total_frames=total_frames,
                    value_from=0,
                    value_to=1,
                    **conf,
                )
                self.transform.rotate(factor * 2 * math.pi)

    def add_orbit_animation(
        self,
        radius: float,
        n_repetitions: int = 1,
        envelope: callable = beating,
        method: str = "linear",
        bpm: float = None,
        beat_bumpyness: float = 0.5,
    ):
        self.animations["orbit"] = dict(
            radius=radius,
            n_repetitions=n_repetitions,
            envelope=envelope,
            method=method,
            bpm=bpm,
            beat_bumpyness=beat_bumpyness,
        )

    def add_scale_animation(
        self,
        factor,
        n_repetitions: int = 1,
        bpm: float = None,
        beat_bumpyness: float = 0.5,
        method: str = "sinusoidal",
        envelope: callable = beating_up_down,
    ):
        self.animations["scale"] = dict(
            value_from=1.0,
            value_to=factor,
            n_repetitions=n_repetitions,
            bpm=bpm,
            beat_bumpyness=beat_bumpyness,
            method=method,
            envelope=envelope,
        )

    def add_attr_animation(
        self,
        attribute,
        target: float,
        n_repetitions: int = 1,
        bpm: float = None,
        beat_bumpyness: float = 0.5,
        method: str = "sinusoidal",
        envelope: callable = beating_up_down,
    ):
        if "attrib" not in self.animations:
            self.animations["attrib"] = dict()
        self.animations["attrib"][attribute] = dict(
            value_to=target,
            n_repetitions=n_repetitions,
            bpm=bpm,
            beat_bumpyness=beat_bumpyness,
            envelope=envelope,
            method=method,
        )

    def add_translation_animation(
        self,
        target: Transform,
        n_repetitions: int = 1,
        bpm: float = None,
        beat_bumpyness: float = 0.5,
        method: str = "sinusoidal",
        envelope: callable = beating_up_down,
    ):
        self.animations["translate"] = dict(
            value_to=target,
            n_repetitions=n_repetitions,
            method=method,
            envelope=envelope,
            bpm=bpm,
            beat_bumpyness=beat_bumpyness,
        )

    def add_rotation_animation(
        self,
        n_rotations: int = 1,
        bpm: float = None,
        beat_bumpyness: float = 0.5,
        method: str = "linear",
        envelope: callable = beating,
    ):
        if n_rotations < 1:
            logger.warn(
                "illegal number of rotations: %d - not adding animation.", n_rotations
            )
            return
        self.animations["rotate"] = dict(
            n_repetitions=n_rotations,
            bpm=bpm,
            beat_bumpyness=beat_bumpyness,
            method=method,
            envelope=envelope,
        )

    def __repr__(self):
        return ET.tostring(self.to_element(), encoding="utf-8").decode()
