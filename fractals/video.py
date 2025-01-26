from __future__ import annotations
import math
import os
import re
import subprocess as sp
import sys
import xml.etree.ElementTree as ET
from glob import glob
from typing import List

from fractals.flame import Flame
from fractals.utils import logger
import json
import shutil
from datetime import datetime


class Video:
    def __init__(
        self,
        flames: List[Flame],
        quality: int = 1000,
        super_sample: int = 2,
        draft: bool = False,
        fps=60,  # reasonable
    ):
        self.flames = flames
        if len(self.flames) == 0:
            raise Exception("Empty flames!")
        self.name = flames[0].element.attrib["name"][6:]
        self.index = int(flames[0].element.attrib["name"][0:3])

        self.fps = fps

        if draft:
            logger.info("DRAFT mode is on. Reduced image size and quality.")
            self.quality = 10
            self.super_sample = 0
            time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
            self.directory = f"{self.index:03d} - draft {time}"
        else:
            self.quality = quality
            self.super_sample = super_sample
            self.directory = f"{self.index:03d} - {self.name}"

        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

    @classmethod
    def from_animation(
        cls, flame: Flame, total_frames: int, **kwargs
    ) -> Video:
        return Video(flames=flame.animate(total_frames), **kwargs)

    @classmethod
    def from_animated_flame_file(cls, animation_file: str, **kwargs) -> Video:
        root = ET.parse(animation_file).getroot()
        flames: List[Flame] = [Flame.from_element(el) for el in root]
        return Video(flames=flames, **kwargs)

    def join(self, name) -> str:
        return os.path.join(self.directory, name)

    @property
    def flame_file(self) -> str:
        return self.join("images/animation.flame")

    @property
    def final_video_file(self) -> str:
        return self.join(f"final_{self.index:03d}.mp4")

    @property
    def preview_video_file(self) -> str:
        return self.join(f"preview_{self.index:03d}.mp4")

    @property
    def image_directory(self):
        return self.join("images")

    def write_file(self):
        root = ET.Element("flames")
        for f in self.flames:
            root.append(f.to_element())
        logger.info(
            "writing %d flames to animation file %s ",
            len(self.flames),
            self.flame_file,
        )
        ET.ElementTree(root).write(self.flame_file)

    def get_last_rendered_flame(self) -> int:
        """
        finds the last flame in the animation file for which there is a PNG
        file.
        """
        pngs = sorted(glob(self.image_directory + "/*.png"))
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
                    return idx + 1
            logger.debug(
                "did not find the last flame id (%s) in the names of flames in this animation",
                last_flame_id,
            )
            logger.debug([f.element.attrib["time"] for f in self.flames])
        else:
            logger.debug(
                "did not find a match for pattern %s in %s",
                pattern,
                last_png_name,
            )
        sys.exit(1)

    def deploy(self) -> None:
        self.render_images()
        self.render_movie()
        self.render_preview()
        self.copy_to_server()

    def copy_to_server(self):
        base_dir = "/home/benjamin/git/wallpaper-server/website"
        index = f"{base_dir}/index.json"
        with open(index) as idx_fh:
            entries = json.load(idx_fh)
        if self.index in [entry["id"] for entry in entries]:
            logger.warn(
                "did not copy files to server, entry for that index "
                "already exists."
            )
            return
        entries.append(
            {
                "id": self.index,
                "previewUrl": os.path.basename(self.preview_video_file),
                "finalUrl": os.path.basename(self.final_video_file),
                "name": self.name,
            }
        )

        with open(index, "w") as out_fh:
            json.dump(
                sorted(entries, key=lambda entry: entry["id"]),
                out_fh,
                indent=2,
            )

        shutil.copy2(self.preview_video_file, base_dir)
        shutil.copy2(self.final_video_file, base_dir)

    def render_images(self) -> None:
        if not os.path.isdir(self.image_directory):
            os.makedirs(self.image_directory)
        self.write_file()

        begin = self.get_last_rendered_flame()
        if begin == len(self.flames) - 1:
            logger.debug("skipping rendering, all pngs are there")
            return
        command = [
            "emberanimate",
            "--opencl",
            "--in",
            os.path.basename(self.flame_file),  # see cwd of Popen cmd below
            "--begin",
            str(begin),
            "--quality",
            str(self.quality),
            "--supersample",
            str(self.super_sample),
        ]

        logger.info("rendering flames of %s", self.flame_file)
        logger.debug("command used for rendering: \n\n%s\n", " ".join(command))
        sp.Popen(command, cwd=self.image_directory).communicate()

    def render_movie(self) -> None:
        pattern = (
            self.image_directory
            + r"/%0"
            + str(
                math.floor(math.log10(self.get_last_rendered_flame() + 1)) + 1
            )
            + "d.png"
        )
        command = [
            "ffmpeg",
            "-framerate",
            f"{self.fps}",  # this is needed once for input
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            f"{pattern}",
            "-c:v",
            "h264_nvenc",
            "-b:v",
            "5M",
            self.final_video_file,
        ]
        logger.info("combining pngs to mp4 file: %s", self.final_video_file)
        sp.Popen(command).communicate()

    def render_preview(self, crf=27) -> None:
        command = [
            "ffmpeg",
            "-i",
            f"{self.final_video_file}",
            "-c:v",
            "libx265",
            "-filter:v",
            "crop=1200:1200:0:400, scale=300:300",
            "-ss",
            "0",
            "-t",  # only 5 seconds
            "5",
            "-crf",
            f"{crf}",
            "-r",  # reduce framerate to 30
            "30",
            self.preview_video_file,
        ]
        logger.info("rendering preview %s", f"{self.preview_video_file}")
        sp.Popen(command).communicate()
