import math
import os
import re
import subprocess as sp
import sys
import xml.etree.ElementTree as ET
from glob import glob
from typing import List

from tqdm import tqdm

from fractals.utils import logger


class Animation:
    def __init__(
        self,
        flames,
        directory: str,
        quality: int = 1000,
        supersample: int = 2,
        movie_file_name: str = "animation.mp4",
        draft: bool = False,
        one_file_per_flame: bool = False,
    ):
        self.flames = flames
        self.directory = directory
        self.filename = os.path.join(self.directory, "animation.flame")
        self.moviename = os.path.join(self.directory, movie_file_name)
        self.one_file_per_flame = one_file_per_flame

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
        if self.one_file_per_flame:
            for i, f in enumerate(self.flames):
                fn = self.directory + f"/{i:05d}.flame"
                if os.path.exists(fn):
                    continue
                root = ET.Element("flames")
                root.append(f.to_element())
                ET.ElementTree(root).write()
        else:
            root = ET.Element("flames")
            for f in self.flames:
                root.append(f.to_element())
            logger.info("writing animation file %s", self.filename)
            ET.ElementTree(root).write(self.filename)

    def get_last_rendered_flame(self) -> int:
        pngs = sorted(glob(self.directory + "/*.png"))
        if not pngs:
            return 0
        last_png_name = os.path.basename(pngs[-1])
        pattern = r"(?P<flame_id>\d*).png"
        match = re.match(pattern, last_png_name)
        if match:
            last_flame_id = match["flame_id"]
            # check if last_flame_id is actually in the flame file
            for idx, f in enumerate(self.flames):
                if int(f.element.attrib["time"]) == int(last_flame_id):
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

    def render(self, verbose: bool = False):

        self.write_file()

        if self.one_file_per_flame:
            finished_pngs = sorted(glob(self.directory + "/*.png"))
            yet_to_be_renderd = [
                self.directory + f"/{i:05d}.flame"
                for i in range(len(self.flames))
                if self.directory + f"/{i:05d}.png" not in finished_pngs
            ]

            logger.info("found %d files to be rendered.", len(yet_to_be_renderd))

            for filename in tqdm(yet_to_be_renderd):
                command = [
                    "emberrender",
                    "--opencl",
                    "--in",
                    filename,
                    "--out",
                    filename[:-5] + "png",
                    "--quality",
                    str(self.quality),
                    "--supersample",
                    str(self.supersample),
                ]
                sp.check_output(command)
            logger.info("all done rendering pngs!")
        else:
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
        if self.one_file_per_flame:
            pattern = r"%05d.png"
        else:
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
