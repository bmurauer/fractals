#!/usr/bin/env python3

import click
from fractals.flame import Flame
from render import get_flame_from_file


@click.command(help="creates an animation where the palette is rotated fully one time.")
@click.argument("inputfile")
@click.option(
    "-fn", "--flame-name", help="name of the flame in the input file to be rotated."
)
@click.option(
    "-fi",
    "--flame-index",
    help="index of the flame from the input file to be rotated. If both --flame-name "
    "and --flame-index are provided, flame-name will be used.",
)
@click.option(
    "-f",
    "--frames",
    default=1000,
    help="how many frames to render. The more frames, the slower the animation.",
)
def colorrotate(
    inputfile: str,
    flame_index: int,
    flame_name: str,
    frames: int,
):
    xml = get_flame_from_file(inputfile, flame_name, flame_index)
    flame = Flame.from_element(xml)
    flames = flame.rotate_colors(frames)
    flames.render()
    flames.convert_to_movie()


if __name__ == "__main__":
    colorrotate()
