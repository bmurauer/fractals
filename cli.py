#!/usr/bin/env python3

import argparse
import logging
import random
import sys
import xml.etree.ElementTree as ET
from flame import Flame

from utils import logger


def get_flame_from_file(file_name: str, flame_name: str, flame_idx: int) -> ET.Element:
    root = ET.parse(file_name).getroot()

    if flame_name:
        flames = [f for f in root if f.attrib["name"] == flame_name]
        if not flames:
            raise Exception(
                "Could not find flame with name {} in file {}."
                "Available flame names:\n{}".format(
                    flame_name,
                    file_name,
                    [f"\n\t{f.attrib['name']}" for f in root]
                ))
        if len(flames) > 1:
            raise Exception("More than one flame with name {} in file {}".format(
                flame_name, file_name))

        return flames[0]
    elif flame_idx:
        return root[flame_idx]
    else:
        return random.choice(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("method")
    parser.add_argument("inputfile")
    parser.add_argument("-fi", "--flame-index", type=int, default=0)
    parser.add_argument("-fn", "--flame-name")
    parser.add_argument("-s", "--steps", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-i", "--interpframes", type=int, default=125,
                        help="how many frames per step")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.method == "mutate":
        raise NotImplementedError()
    elif args.method == "interpolate":
        raise NotImplementedError()
    elif args.method == "colorrotate":
        xml = get_flame_from_file(args.inputfile, args.flame_name, args.flame_index)
        flame = Flame.from_element(xml)
        flames = flame.rotate_colors(args.interpframes)
        flames.render()
        flames.convert_to_movie()
    else:
        logger.error('unknown method: %s', args.method)
        sys.exit(1)
