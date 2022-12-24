from __future__ import annotations

import copy
import datetime
import math
import os
import random
import re
import subprocess as sp
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from glob import glob
from textwrap import wrap
from typing import List, Union

import numpy as np

from fractals.utils import logger

FPS = 30


# def interpolate_linear(
#     value_from: Union[Transform, float],
#     value_to: Union[Transform, float],
#     n_repetitions: int,
#     frame: int,
#     total_frames: int,
#     bpm: float = None,
#     bumpyness: float = 0.5,
# ):
#     """
#     Splits total_frames into n_repetitions sections.
#     Each section is then split into an "ascending" and "descending" half.
#     """

#     frames_per_repetition = total_frames / n_repetitions
#     current_repetition = frame // frames_per_repetition
#     frame_in_repetition = frame % frames_per_repetition

#     factor = beating_interpolation(
#         frame_in_repetition, frames_per_repetition, bpm, bumpyness
#     )
#     diff = value_to - value_from

#     if (frame % frames_per_repetition) <= frames_per_repetition / 2:
#         # ascending half
#         return value_from + diff * 2 * factor
#     else:
#         # descending half
#         return value_from - diff * 2 * factor + diff * 2


# def interpolate_sinusoidal(
#     value: float,
#     offset: float,
#     frame: int,
#     n_repetitions: int,
#     total_frames: int,
#     bpm: float = None,
#     bumpyness: float = 0.5,
# ):
#     # we need a function that behaves like a sinus to smoothly transition from 0 to 1
#     def smooth(x):
#         return (math.cos(math.pi * (2 * x + 1)) + 1) / 2

#     frames_per_repetition = total_frames / n_repetitions
#     frame_in_repetition = frame % frames_per_repetition
#     # no check necessary, the cosine function automagically has a bump back to 0.
#     return value + offset * smooth(frame_in_repetition / frames_per_repetition)


def rotate(
    value: Transform,
    n_rotations: int,
    frame: int,
    total_frames: int,
    bpm: float = None,
    bumpyness: float = 0.5,
):

    factor = repeat_beating_up(
        0, 1, frame, total_frames, n_rotations, "sigmoid", bpm, bumpyness
    )

    value.rotate(factor * 2 * math.pi)
    return value


def ramp_sinusoidal(frame, total_frames):
    return (math.cos(math.pi * ((frame / total_frames) + 1)) + 1) / 2


def ramp_linear(frame, total_frames):
    return frame / total_frames


def ramp_sigmoid(frame, total_frames, bumpyness):
    linear = ramp_linear(frame, total_frames)
    sigmoid = 2 / (
        1 + math.pow(math.e, -((10 / total_frames) * (frame - total_frames)))
    )
    return sigmoid * bumpyness + linear * (1 - bumpyness)


def ramp_inverse_sigmoid(frame, total_frames, bumpyness):
    linear = ramp_linear(frame, total_frames)
    inv_sigmoid = 2 / (1 + math.pow(math.e, -((10 / total_frames) * (frame)))) - 1
    return inv_sigmoid * bumpyness + linear * (1 - bumpyness)


def repeat_smooth_up(
    value_from, value_to, frame, total_frames, n_repetitions, method, bumpyness=0.5
):
    frames_per_repetition = total_frames / n_repetitions
    current_repetition = frame // frames_per_repetition
    frame_in_repetition = frame % frames_per_repetition

    factor = None
    if method == "linear":
        factor = ramp_linear(frame_in_repetition, frames_per_repetition)
    elif method == "sinusoidal":
        factor = ramp_sinusoidal(frame_in_repetition, frames_per_repetition)
    elif method == "sigmoid":
        factor = ramp_sigmoid(frame_in_repetition, frames_per_repetition, bumpyness)
    else:
        raise Exception("unknown method: " + method)

    diff = value_to - value_from
    return value_from + diff * factor


def repeat_beating_up(
    value_from, value_to, frame, total_frames, method, bpm, bumpyness=0.5
):
    diff = value_to - value_from
    frames_per_beat = FPS / (bpm / 60)

    if total_frames % frames_per_beat != 0:
        logger.error(
            "Can't subdivide %d frames into beats of %d frames",
            total_frames,
            frames_per_beat,
        )
        sys.exit(1)

    n_beats: int = total_frames // frames_per_beat
    # we are in this beat right now:
    current_beat = frame // frames_per_beat
    # within the beat, we are currently at frame:
    frame_in_beat = frame % frames_per_beat

    beat_start_frame = current_beat * frames_per_beat
    beat_end_frame = (current_beat + 1) * frames_per_beat - 1

    start_value = repeat_smooth_up(0, 1, beat_start_frame, total_frames, 1, method)
    end_value = repeat_smooth_up(0, 1, beat_end_frame, total_frames, 1, method)

    bumpy_factor = repeat_smooth_up(
        start_value,
        end_value,
        frame_in_beat,
        frames_per_beat,
        1,
        "sigmoid",
        bumpyness=1.0,
    )

    result = value_from + diff * bumpy_factor

    print(
        frame_in_beat,
        current_beat,
        beat_start_frame,
        beat_end_frame,
        start_value,
        end_value,
        bumpy_factor,
        result,
    )
    return result


def repeat_smooth_up_down(
    value_from, value_to, frame, total_frames, n_repetitions, method, bumpyness=0.5
):

    frames_per_half = total_frames // 2
    frame_in_half = frame % frames_per_half

    if frame < frames_per_half:
        return repeat_smooth_up(
            value_from, value_to, frame_in_half, frames_per_half, n_repetitions, method
        )
    else:
        return repeat_smooth_up(
            value_to, value_from, frame_in_half, frames_per_half, n_repetitions, method
        )


def repeat_beating_up_down(
    value_from: Union[Transform, float],
    value_to: Union[Transform, float],
    frame,
    total_frames,
    n_repetitions,
    method,
    bpm,
    bumpyness,
):

    frames_per_half = total_frames // 2
    frame_in_half = frame % frames_per_half

    if frame < frames_per_half:
        return repeat_beating_up(
            value_from,
            value_to,
            frame_in_half,
            frames_per_half,
            method,
            bpm,
            bumpyness,
        )
    else:
        return repeat_beating_up(
            value_to,
            value_from,
            frame_in_half,
            frames_per_half,
            method,
            bpm,
            bumpyness,
        )


# def beating_interpolation(frame, total_frames, bpm, bumpyness):

#     """
#     Returns a smooth sigmoid function ramping up to the desired end value.
#     """

#     linear = frame / total_frames
#     if not bpm:
#         return linear

#     # e.g., bpm=120 means two beats per second, means 15 frames per beat
#     frames_per_beat = FPS / (bpm / 60)

#     if total_frames % frames_per_beat != 0:
#         logger.error(
#             "Can't subdivide %d frames into beats of %d frames",
#             total_frames,
#             frames_per_beat,
#         )
#         sys.exit(1)

#     n_beats: int = total_frames // frames_per_beat
#     # we are in this beat right now:
#     beat_idx = frame // frames_per_beat
#     # within the beat, we are currently at frame:
#     frame_in_beat = frame % frames_per_beat

#     beat_start_frame = beat_idx * frames_per_beat
#     beat_end_frame = (beat_idx + 1) * frames_per_beat - 1
#     #  start and end of each beat are linear
#     start_value = beat_start_frame / total_frames
#     end_value = beat_end_frame / total_frames

#     a = 10 / frames_per_beat
#     sigmoid = (
#         2
#         * (end_value - start_value)
#         / (1 + math.pow(math.e, -(a * (frame_in_beat - frames_per_beat))))
#         + start_value
#     )
#     return sigmoid * bumpyness + linear * (1 - bumpyness)


def rotate_vector(vector, rad, rotation_center=None):
    if rotation_center is None:
        rotation_center = np.array([0, 0, 1]).T
    x = np.array(
        [
            [math.cos(rad), -math.sin(rad), 0],
            [math.sin(rad), math.cos(rad), 0],
            [0, 0, 1],
        ]
    ).dot(vector - rotation_center)
    return x + rotation_center


def scale(
    value: Transform,
    factor: float,
    n_repetitions: int,
    frame: int,
    total_frames: int,
    bpm: float = None,
    bumpyness: float = 0.5,
):

    target = deepcopy(value)

    scaled_factor = repeat_beating_up_down(
        1, factor, frame, total_frames, n_repetitions, "sinusoidal", bpm, bumpyness
    )

    target.coefs[0][0] *= scaled_factor
    target.coefs[1][1] *= scaled_factor

    return target


def orbit(
    value: Transform,
    radius: float,
    n_rotations: int,
    frame: int,
    total_frames: int,
):
    x = np.array([radius, 0, 1]).T
    x = rotate_vector(x, 2 * math.pi * n_rotations * frame / total_frames)
    x[0] -= radius
    value.translate(x[0], x[1])
    return value


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

    def translate_orbit_step(self, radius, frame, total_frames):
        x = np.array([radius, 0, 1]).T
        x = rotate_vector(x, 2 * math.pi * frame / total_frames)
        self.translate(x[0] - radius, x[1])

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
        t.coefs[2] = np.array([1, 1, 1])
        return t

    def __sub__(self, other):
        t = deepcopy(self)
        t.coefs = self.coefs - other.coefs
        t.coefs[2] = np.array([1, 1, 1])
        return t

    def __truediv__(self, other: Union[float, int]):
        t = deepcopy(self)
        t.coefs = self.coefs / other
        t.coefs[2] = np.array([1, 1, 1])
        return t

    def __mul__(self, other: Union[float, int]):
        t = deepcopy(self)
        t.coefs = self.coefs * other
        t.coefs[2] = np.array([1, 1, 1])
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
    ):
        self.count = count
        self.format = format
        self.colors = colors
        self.original = deepcopy(colors)

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

    def animate(self, n_rotations: int, frame: int, total_frames: int) -> None:
        frames_per_rotation = total_frames / n_rotations
        frame_in_rotation = frame % frames_per_rotation
        self.colors = self.rotate(frame_in_rotation / frames_per_rotation)

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
        color: float = 1.0,
        weight: float = 1.0,
    ):
        self.element = element
        self.coefs = coefs
        self.color = color
        self.weight = weight
        self.animations = dict()

    @classmethod
    def from_element(cls, element: ET.Element):
        coefs = Transform(element.attrib["coefs"])
        color = float(element.attrib["color"])
        # final xforms don't have weight
        if "weight" in element.attrib:
            weight = float(element.attrib["weight"])
        else:
            weight = 1.0
        return XForm(element, coefs, color, weight)

    def to_element(self) -> ET.Element:
        self.element.attrib["color"] = str(self.color)
        self.element.attrib["coefs"] = str(self.coefs)
        self.element.attrib["weight"] = str(self.weight)
        return self.element

    def animate(self, frame, total_frames):
        if "attrib" in self.animations:
            for name, properties in self.animations["attrib"].items():
                target = properties["target"]
                n_reps = properties["n_repetitions"]
                bpm = properties["bpm"]
                bumpyness = properties["bumpyness"]
                old = float(self.element.attrib[name])
                offset = target - old
                value_as_str = str(
                    round(
                        repeat_smooth_up_down(
                            old,
                            target,
                            frame,
                            total_frames,
                            n_repetitions,
                            "sinusoidal",
                        ),
                        7,
                    )
                )
                self.element.attrib[name] = value_as_str

        if "orbit" in self.animations:
            radius = self.animations["orbit"]["radius"]
            n_reps = self.animations["orbit"]["n_repetitions"]
            self.coefs = orbit(self.coefs, radius, n_reps, frame, total_frames)

        if "scale" in self.animations:
            sconf = self.animations["scale"]
            self.coefs = scale(
                self.coefs,
                sconf["factor"],
                sconf["n_repetitions"],
                frame,
                total_frames,
                sconf["bpm"],
                sconf["bumpyness"],
            )

        if "translate" in self.animations:
            anim_conf = self.animations["translate"]
            # self.coefs = interpolate_linear(
            #     self.coefs,
            #     anim_conf["target"],
            #     anim_conf["n_repetitions"],
            #     frame,
            #     total_frames,
            #     anim_conf["bpm"],
            #     anim_conf["bumpyness"],
            # )
            self.coefs = repeat_smooth_up_down(
                self.coefs,
                anim_conf["target"],
                frame,
                total_frames,
                anim_conf["n_repetitions"],
                "sinusoidal",
            )

        if "rotation" in self.animations:
            n_rotations = self.animations["rotation"]["n_rotations"]
            self.coefs = rotate(
                self.coefs,
                n_rotations,
                frame,
                total_frames,
                self.animations["rotation"]["bpm"],
                self.animations["rotation"]["bumpyness"],
            )

    def add_orbit_animation(self, radius, n_repetitions: int = 1, bpm: float = None):
        self.animations["orbit"] = dict(
            radius=radius, n_repetitions=n_repetitions, bpm=bpm
        )

    def add_scale_animation(
        self, factor, n_repetitions: int = 1, bpm: float = None, bumpyness: float = 0.5
    ):
        self.animations["scale"] = dict(
            factor=factor, n_repetitions=n_repetitions, bpm=bpm, bumpyness=bumpyness
        )

    def add_attr_animation(
        self,
        attribute,
        target: float,
        n_repetitions: int = 1,
        bpm: float = None,
        bumpyness: float = 0.5,
    ):
        if "attrib" not in self.animations:
            self.animations["attrib"] = dict()
        self.animations["attrib"][attribute] = dict(
            target=target, n_repetitions=n_repetitions, bpm=bpm, bumpyness=bumpyness
        )

    def add_translation_animation(
        self,
        target: Transform,
        n_repetitions: int = 1,
        bpm: float = None,
        bumpyness: float = 0.5,
    ):
        self.animations["translate"] = dict(
            target=target, n_repetitions=n_repetitions, bpm=bpm, bumpyness=bumpyness
        )

    def add_rotation_animation(
        self, n_rotations: int = 1, bpm: float = None, bumpyness: float = 0.5
    ):
        self.animations["rotation"] = dict(
            n_rotations=n_rotations, bpm=bpm, bumpyness=bumpyness
        )

    def __repr__(self):
        return ET.tostring(self.to_element(), encoding="utf-8").decode()


def rotate_flame(old_value, n_rotations, frame, total_frames) -> str:
    new_value = 360 * n_rotations * frame / total_frames % 360
    return str(round(new_value, 4))


class Flame:
    def __init__(
        self,
        element: ET.Element,
        palette: Palette,
        xforms: List[XForm],
        final_xform: XForm,
        draft: bool = False,
    ):
        self.element: ET.Element = element
        self.palette: Palette = palette
        self.xforms: List[XForm] = xforms
        self.final_xform: XForm = final_xform
        self.animations = {}
        self.draft = draft

        if draft:
            forced_image_size = "800 600"
        else:
            forced_image_size = "4096 2160"
        if element.attrib["size"] != forced_image_size:
            logger.warn(
                "overwriting size %s to %s", element.attrib["size"], forced_image_size
            )
            self.element.attrib["size"] = forced_image_size

    @classmethod
    def from_element(cls, element: ET.Element, draft: bool = False) -> Flame:
        xforms = [XForm.from_element(xform) for xform in element.findall("xform")]
        final_xform: XForm = None
        if element.find("finalxform") is not None:
            final_xform = XForm.from_element(element.find("finalxform"))
        return Flame(
            element,
            Palette.from_element(element.find("palette")),
            xforms,
            final_xform,
            draft=draft,
        )

    def to_element(self) -> ET.Element:
        clone = copy.deepcopy(self.element)
        clone[:] = []
        [clone.append(xform.to_element()) for xform in self.xforms]
        if self.final_xform:
            clone.append(self.final_xform.to_element())
        clone.append(self.palette.to_element())
        return clone

    def add_rotation_animation(self, n_rotations: int = 1, bpm: float = None):
        self.animations["rotation"] = dict(n_rotations=n_rotations, bpm=bpm)

    def add_palette_rotation_animation(self, n_rotations: int = 1, bpm: float = None):
        self.animations["palette"] = dict(n_rotations=n_rotations, bpm=bpm)

    def animate(self, total_frames):
        result: List[Flame] = []
        for frame in range(total_frames):
            clone = copy.deepcopy(self)
            clone.element.attrib["time"] = str(frame + 1)
            if "rotation" in self.animations:
                old = float(self.element.attrib["rotate"])
                n_rotations = self.animations["rotation"]["n_rotations"]
                self.element.attrib["rotate"] = rotate_flame(
                    old, n_rotations, frame, total_frames
                )
            if "palette" in self.animations:
                n_rotations = self.animations["palette"]["n_rotations"]
                clone.palette.animate(n_rotations, frame, total_frames)
            for xform in clone.xforms:
                xform.animate(frame, total_frames)
            result.append(clone)
        draft = "_draft" if self.draft else ""
        return Flames(
            result,
            f'color-{self.element.attrib["name"]}-{get_time()}',
            movie_file_name="_" + self.element.attrib["name"] + draft + ".mp4",
            draft=self.draft,
        )

    def __repr__(self):
        return ET.tostring(self.to_element()).decode()


class Flames:
    def __init__(
        self,
        flames: List[Flame],
        directory: str,
        quality: int = 1000,
        supersample: int = 2,
        movie_file_name: str = "animation.mp4",
        draft: bool = False,
    ):
        self.flames = flames
        self.directory = directory
        self.filename = os.path.join(self.directory, "animation.flame")
        self.moviename = os.path.join(self.directory, movie_file_name)

        if draft:
            logger.info("DRAFT mode is on. Requced image size and quality.")
            self.quality = 100
            self.supersample = 1
        else:
            self.quality = quality
            self.supersample = supersample

    def write_file(self):
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        root = ET.Element("flames")
        for f in self.flames:
            root.append(f.to_element())
        logger.info("writing animation file %s", self.filename)
        ET.ElementTree(root).write(self.filename)

    def get_last_rendered_flame(self) -> int:
        pngs = sorted(glob(self.directory + "/*.png"))
        if not pngs:
            logger.debug("did not find any pngs in %s", self.directory)
            return 0
        last_png_name = os.path.basename(pngs[-1])
        pattern = r"(?P<flame_id>\d*).png"
        match = re.match(pattern, last_png_name)
        if match:
            last_flame_id = match["flame_id"]
            # check if last_flame_id is actually in the flame file
            for idx, f in enumerate(self.flames):
                if f.element.attrib["time"] == last_flame_id:
                    return idx + 1  # the last found flame should not be rendered again
            logger.debug(
                "did not find the last flame id (%s) in the names of flames in this animation",
                last_flame_id,
            )
            logger.debug([f.element.attrib["time"] for f in self.flames])
        else:
            logger.debug(
                "did not find a match for pattern %s in %s", pattern, last_png_name
            )
        sys.exit(1)

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
            str(begin),
            "--quality",
            str(self.quality),
            "--supersample",
            str(self.supersample),
        ]

        logger.info("rendering flames of %s", self.filename)
        logger.debug("command used for rendering: \n\n%s\n", " ".join(command))
        sp.Popen(command).communicate()

    def convert_to_movie(self):
        pattern = (
            r"%0"
            + str(math.floor(math.log10(self.get_last_rendered_flame())) + 1)
            + "d.png"
        )
        command = [
            "ffmpeg",
            "-i",
            f"{os.path.join(self.directory, pattern)}",
            "-i",
            "logo/logo.png",
            "-filter_complex",
            "[1]format=rgba,colorchannelmixer=aa=0.2[logo];[0][logo]overlay=W-w-20:H-h-20:format=auto,format=yuv420p",
            "-r",
            "25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "22",
            self.moviename,
        ]
        logger.info("combining pngs to mp4 file: %s", self.moviename)
        sp.Popen(command).communicate()
