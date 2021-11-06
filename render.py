#!/usr/bin/env python3
import argparse
import copy
import datetime
import logging
import math
import os
import random
import re
import shutil
import subprocess as sp
import sys
import tempfile
import xml.etree.ElementTree as ET
from glob import glob
from textwrap import wrap
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("Logger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y%m%d-%H:%M:%S"
)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)


def get_last_rendered_flame(root: ET.Element, sq_dir: str) -> int:
    pngs = sorted(glob(sq_dir + "/*.png"))
    if not pngs:
        return 0
    last_png_name = os.path.basename(pngs[-1])
    pattern = r"(?P<flame_id>\d*).png"
    match = re.match(pattern, last_png_name)
    if match:
        last_flame_id = match["flame_id"]
        # check if last_flame_id is actually in the flame file
        for idx, f in enumerate(root):
            if f.attrib["name"] == last_flame_id:
                return idx + 1  # the last found flame should not be rendered again
    return 0


def create_keyframes_file(
    file_name: str,
    flames: List[ET.Element],
) -> None:
    collection = ET.Element("flames")
    [collection.append(f) for f in flames]
    logger.debug("creating keyframes file: %s", file_name)
    ET.ElementTree(collection).write(file_name)


def create_animation_file(
    keyframes_filename: str, animation_filename: str, interpframes: int
) -> None:
    generate_parameters = {
        "--sequence": keyframes_filename,
        "--interpframes": str(interpframes),
        "--interploops": "0",
        "--loops": "0",
        "--loopframes": "0",
        "--stagger": "0.1",
    }
    generate_command = ["embergenome"] + [
        f"{k}={v}" for k, v in generate_parameters.items()
    ]
    with open(animation_filename, "w") as o_f:
        logger.debug("creating animation file: %s", animation_filename)
        sp.Popen(generate_command, stdout=o_f).communicate()


def render_flames(animation_filename: str, sq_dir: str) -> None:
    animation_flames = ET.parse(animation_filename).getroot()
    begin = get_last_rendered_flame(animation_flames, sq_dir)
    if begin == len(animation_flames) - 1:
        logger.debug("skipping rendering, all pngs are there")
        return

    flags = [
        "--opencl",
    ]
    kwargs = {
        "--in": animation_filename,
        "--begin": begin,
        "--quality": 5000,
        "--supersample": 4,
    }
    animate_parameters = flags + [f"{k}={v}" for k, v in kwargs.items()]
    animate_command = ["emberanimate"] + animate_parameters
    logger.debug("rendering flames of %s", animation_filename)
    logger.debug("command used for rendering: %s", str(animate_command))
    sp.Popen(animate_command, stdout=sp.PIPE).communicate()


def interpolate_flame_collection(input_file: str, interpframes: int) -> None:
    root = ET.parse(input_file).getroot()
    for i in range(len(root) - 1):
        f0, f1 = root[i], root[i + 1]
        interpolate_flames(f0, f1, interpframes)


def interpolate_flames(
    f0: ET.Element, f1: ET.Element, interpframes: int, directory=None
):
    assert f0.attrib["size"] == f0.attrib["size"]
    sq_dir = directory or f'seq-{f0.attrib["name"]}-{f1.attrib["name"]}'
    if not os.path.isdir(sq_dir):
        os.makedirs(sq_dir)
    keyframes_filename = os.path.join(sq_dir, "keyframes.flame")
    if not os.path.isfile(keyframes_filename):
        create_keyframes_file(keyframes_filename, [f0, f1])
    animation_filename = os.path.join(sq_dir, "animation.flame")
    if not os.path.isfile(animation_filename):
        create_animation_file(keyframes_filename, animation_filename, interpframes)
    render_flames(animation_filename, sq_dir)


def mutate_float(value_as_str: str, mutation_range: Tuple[float, float]) -> str:
    return str(float(value_as_str) + random.uniform(*mutation_range))


def rotate(coefs: np.ndarray, deg: float) -> np.ndarray:
    rad = math.pi * deg / 180.0
    return coefs.dot(
        np.array(
            [
                [math.cos(rad), -math.sin(rad), 0],
                [math.sin(rad), math.cos(rad), 0],
                [0, 0, 1],
            ]
        )
    )


def scale(coefs: np.ndarray, factor) -> np.ndarray:
    return coefs.dot(np.array([[factor, 0, 0], [0, factor, 0], [0, 0, 1]]))


def translate(coefs: np.ndarray, x, y) -> np.ndarray:
    return coefs.dot(
        np.array(
            [
                [1, 0, x],
                [0, 1, y],
                [0, 0, 1],
            ]
        )
    )


def mutate_coefs(
    value: str, coef_scale_range, coef_rotation_range, coef_translation_range
) -> str:
    coefficients = np.array([float(v) for v in value.split(" ")]).reshape(3, 2).T
    coefficients = translate(
        coefficients,
        random.uniform(*coef_translation_range),
        random.uniform(*coef_translation_range),
    )
    coefficients = scale(coefficients, random.uniform(*coef_scale_range))
    coefficients = rotate(coefficients, random.uniform(*coef_rotation_range))
    return " ".join([str(c)[:6] for c in coefficients.T.flatten()])


def mutate_flame(
    original: ET.Element,
    coef_translation_range=(-0.1, 1.0),
    coef_rotation_range=(0, 90),
    coef_scale_range=(0.9, 1.1),
    post_translation_range=(-0.1, 1.0),
    post_rotation_range=(0, 90),
    post_scale_range=(0.9, 1.1),
    color_range=(0, 1),
    variation_range=(-0.1, 0.1),
    variation_range_map=None,
) -> ET.Element:
    mutation = copy.deepcopy(original)
    for xform in mutation.findall("xform")[-1:]:
        ignored_properties = [
            "var_color",
            "symmetry",
            "name",
            "animate",
            "opacity",
            "coefs",  # special format
            "post",
        ] + [
            k for k in xform.attrib.keys() if k.endswith("_power")
        ]  # integers
        xform.attrib["coefs"] = mutate_coefs(
            xform.attrib["coefs"],
            coef_scale_range,
            coef_rotation_range,
            coef_translation_range,
        )
        if "post" in xform.attrib.keys():
            xform.attrib["post"] = mutate_coefs(
                xform.attrib["post"],
                post_scale_range,
                post_rotation_range,
                post_translation_range,
            )
        for key, value in xform.attrib.items():
            if key not in ignored_properties:
                var_range = variation_range
                if variation_range_map and key in variation_range_map:
                    var_range = variation_range_map[key]
                xform.attrib[key] = mutate_float(value, var_range)
    return mutation


def animate_mutation_sequence(
    original_flame: ET.Element, steps: int, interpframes: int
):

    # --- STEP 1: preparation stuff
    flame_name = original_flame.attrib["name"]
    original_flame.attrib["name"] = "original"
    original_file = "original.flame"
    collected_file = "collected.flame"
    movie_file = f"mut-{flame_name}.mp4"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    mut_dir = f"mut-{flame_name}-{timestamp}"
    if not os.path.isdir(mut_dir):
        os.makedirs(mut_dir)
    os.chdir(mut_dir)
    if not os.path.isfile(original_file):
        # the embergenome command can't mutate a specific flame, it will always pick a
        # random flame from a file. Therefore, the flame has to be in a separate file.
        ET.ElementTree(original_flame).write(original_file)
    logger.info(
        "animating mutation sequence for flame %s, using %d steps.", flame_name, steps
    )

    # --- STEP 2: generate mutations
    mutations = [original_flame]
    for step in range(steps):
        mutation_flame = mutate_flame(original_flame)
        mutation_flame.attrib["name"] = f"mutation-{step}"
        mutations.append(mutation_flame)
    original_clone = copy.deepcopy(original_flame)
    original_clone.attrib["name"] = "original-1"
    mutations.append(original_clone)
    if not os.path.isfile(collected_file):
        create_keyframes_file(collected_file, mutations)

    # --- STEP 3: render images
    logger.info("rendering png image files")
    interpolate_flame_collection(collected_file, interpframes)

    # --- STEP 4: render mp4 from images
    logger.info(
        "rendering mp4 movie file: %s",
    )
    pngs = []
    pngs += sorted(glob("seq-original-mutation-0/*.png"))
    for step in range(steps - 1):
        pngs += sorted(glob(f"seq-mutation-{step}-mutation-{step + 1}/*.png"))
    pngs += sorted(glob(f"seq-mutation-{steps - 1}-original/*.png"))
    combine_pngs_to_mp4(movie_file, pngs)


def combine_pngs_to_mp4(output_filename: str, pngs: List[str]):
    tmpdir = tempfile.TemporaryDirectory()

    for i, png in enumerate(pngs):
        shutil.copy(png, os.path.join(tmpdir.name, f"{i:04d}.png"))
    command = [
        "ffmpeg",
        "-i",
        f'{ os.path.join(tmpdir.name, r"%04d.png")}',
        "-r",
        "25",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "25",
        output_filename,
    ]
    logger.info("combining pngs to mp4 file: %s", output_filename)
    sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE).communicate()
    shutil.rmtree(tmpdir.name)


def rotate_palette(string: str, steps: int):
    result = "".join([x.strip() for x in string.split("\n")])
    for i in range(steps):
        # one step consists of 6 hex digits (for RGB)
        result = result[6:] + result[:6]
    return "".join([f"\n      {x}" for x in wrap(result, 48)]) + "\n"


def color_rotate(flame: ET.Element, frames: int):

    # --- STEP 1: preparation stuff
    flame_name = flame.attrib["name"]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    animation_dir = f"color-{flame_name}-{timestamp}"
    if not os.path.isdir(animation_dir):
        os.makedirs(animation_dir)
    flame.attrib["time"] = str(0)

    collection = ET.Element("flames")
    collection.append(copy.deepcopy(flame))
    # calculate how much color needs to be added with each frame
    logger.info("creating %d frames for color rotation of %s", frames, flame_name)
    last_rotation = 0
    for i in range(frames):
        clone = copy.deepcopy(flame)
        clone.attrib["time"] = str(i + 1)
        palette = clone.find("palette")

        full_rotation = float(palette.attrib["count"])
        rotation_steps = i * full_rotation / frames
        palette.text = rotate_palette(palette.text, round(rotation_steps))
        collection.append(clone)

    animation_file = os.path.join(animation_dir, "frames.flame")
    logger.debug("creating keyframes file: %s", animation_file)
    ET.ElementTree(collection).write(animation_file)

    logger.info("rendering frames")
    render_flames(animation_file, animation_dir)
    pngs = sorted(glob(animation_dir + "/*.png"))

    logger.info("combining pngs to mp4")
    combine_pngs_to_mp4(os.path.join(animation_dir, f"mut-{flame_name}.mp4"), pngs)


def get_flame_from_file(file_name: str, flame_name: str, flame_idx: int) -> ET.Element:
    root = ET.parse(file_name).getroot()

    if flame_name:
        flames = [f for f in root if f.attrib["name"] == flame_name]
        if not flames:
            raise Exception(
                "Could not find flame with name {} in file {}."
                "Available flame names:\n{}".format(
                    flame_name, file_name, [f"\n\t{f.attrib['name']}" for f in root]
                )
            )
        if len(flames) > 1:
            raise Exception(
                "More than one flame with name {} in file {}".format(
                    flame_name, file_name
                )
            )

        return flames[0]
    elif flame_idx:
        return root[flame_idx]
    else:
        return random.choice(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method")
    parser.add_argument("inputfile")
    parser.add_argument("-fi", "--flame-index", type=int, default=0)
    parser.add_argument("-fn", "--flame-name")
    parser.add_argument("-s", "--steps", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-i", "--interpframes", type=int, default=125, help="how many frames per step"
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.method == "mutate":
        flame = get_flame_from_file(args.inputfile, args.flame_name, args.flame_index)
        animate_mutation_sequence(flame, args.steps, args.interpframes)
    elif args.method == "interpolate":
        interpolate_flame_collection(args.inputfile, args.interpframes)
    elif args.method == "colorrotate":
        flame = get_flame_from_file(args.inputfile, args.flame_name, args.flame_index)
        color_rotate(flame, args.interpframes)
    else:
        logger.error("unknown method: %s", args.method)
        sys.exit(1)
