#!/usr/bin/env python3
import argparse
import os
from glob import glob
import subprocess as sp
import xml.etree.ElementTree as ET
import re
import logging

logger = logging.getLogger('Logger')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                              datefmt='%Y%m%d-%H:%M:%S')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)


def get_last_rendered_flame(root: ET.Element, sq_dir: str) -> int:
    pngs = sorted(glob(sq_dir + '/*.png'))
    if not pngs:
        return 0
    last_png_name = os.path.basename(pngs[-1])
    pattern = r"seq-(\d\d\d)-(\d\d\d)-(?P<flame_id>\d*).png"
    match = re.match(pattern, last_png_name)
    if match:
        last_flame_id = match['flame_id']
        # check if last_flame_id is actually in the flame file
        for idx, flame in enumerate(root):
            if flame.attrib['name'] == last_flame_id:
                return idx + 1  # the last found flame should not be rendered again
    return 0


def create_keyframes_file(
        f0: ET.Element,
        f1: ET.Element,
        keyframes_filename: str,
) -> None:
    flames = ET.Element('flames')
    flames.append(f0)
    flames.append(f1)
    logger.info('creating keyframes file: %s', keyframes_filename)
    ET.ElementTree(flames).write(keyframes_filename)


def create_animation_file(keyframes_filename: str, animation_filename: str) -> None:
    generate_parameters = {
        '--sequence': keyframes_filename,
        '--interpframes': '125',
        '--interploops': '0',
        '--loops': '0',
        '--loopframes': '0',
    }
    generate_command = ['embergenome'] + [f'{k}={v}'
                                          for k, v
                                          in generate_parameters.items()]
    with open(animation_filename, 'w') as o_f:
        logger.info('creating animation file: %s', animation_filename)
        sp.Popen(generate_command, stdout=o_f).communicate()


def main(input_file: str) -> None:
    root = ET.parse(input_file).getroot()
    for i in range(len(root) - 1):
        f0, f1 = root[i], root[i + 1]
        # check sizes

        assert f0.attrib["size"] == f0.attrib["size"] == "2560 1440"

        sq_dir = f'seq-{f0.attrib["name"]}-{f1.attrib["name"]}'
        if not os.path.isdir(sq_dir):
            os.makedirs(sq_dir)

        keyframes_filename = os.path.join(sq_dir, 'keyframes.flame')
        if not os.path.isfile(keyframes_filename):
            create_keyframes_file(f0, f1, keyframes_filename)

        animation_filename = os.path.join(sq_dir, 'animation.flame')
        if not os.path.isfile(animation_filename):
            create_animation_file(keyframes_filename, animation_filename)

        render_flames(animation_filename, sq_dir)


def render_flames(animation_filename: str, sq_dir: str) -> None:
    animation_flames = ET.parse(animation_filename).getroot()
    begin = get_last_rendered_flame(animation_flames, sq_dir)
    if begin == len(animation_flames) - 1:
        logger.debug("skipping rendering, all pngs are there")
        return

    flags = [
        '--opencl',
    ]
    kwargs = {
        '--in': animation_filename,
        '--begin': begin,
        '--quality': 5000,
        '--supersample': 4,
    }
    animate_parameters = flags + [f'{k}={v}' for k, v in kwargs.items()]
    animate_command = ['emberanimate'] + animate_parameters
    sp.Popen(animate_command, shell=False, bufsize=1).communicate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile")
    args = parser.parse_args()
    main(args.inputfile)
