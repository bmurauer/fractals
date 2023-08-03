from __future__ import annotations  # forward type hints

import random
import xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import datetime
from typing import List, Optional

from fractals.video import Video
from fractals.palette import Palette
from fractals.utils import logger
from fractals.xform import XForm


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
            forced_image_size = "800 450"
        else:
            forced_image_size = "4096 2160"
        if element.attrib["size"] != forced_image_size:
            logger.warn(
                "overwriting size %s to %s",
                element.attrib["size"],
                forced_image_size,
            )
            self.element.attrib["size"] = forced_image_size

    @classmethod
    def from_element(cls, element: ET.Element, draft: bool = False) -> Flame:
        xforms = [
            XForm.from_element(xform) for xform in element.findall("xform")
        ]
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

    @classmethod
    def from_file(
        cls,
        file_name: str,
        flame_name: Optional[str] = None,
        flame_idx: Optional[int] = None,
        draft: bool = False,
    ) -> Flame:
        root = ET.parse(file_name).getroot()
        if flame_name:
            flames = [f for f in root if f.attrib["name"] == flame_name]
            if not flames:
                raise Exception(
                    "Could not find flame with name {} in file {}."
                    "Available flame names:\n{}".format(
                        flame_name,
                        file_name,
                        [f"\n\t{f.attrib['name']}" for f in root],
                    )
                )
            if len(flames) > 1:
                raise Exception(
                    "More than one flame with name {} in file {}".format(
                        flame_name, file_name
                    )
                )

            return Flame.from_element(flames[0], draft=draft)
        elif flame_idx:
            return Flame.from_element(root[flame_idx], draft=draft)
        else:
            return Flame.from_element(random.choice(root), draft=draft)

    def to_element(self) -> ET.Element:
        clone = deepcopy(self.element)
        clone[:] = []
        [clone.append(xform.to_element()) for xform in self.xforms]
        if self.final_xform:
            clone.append(self.final_xform.to_element())
        clone.append(self.palette.to_element())
        return clone

    def add_rotation_animation(self, n_rotations: int = 1, bpm: float = None):
        self.animations["rotation"] = dict(n_rotations=n_rotations, bpm=bpm)

    def add_palette_rotation_animation(
        self, n_rotations: int = 1, bpm: float = None
    ):
        self.animations["palette"] = dict(n_rotations=n_rotations, bpm=bpm)

    def animate(self, total_frames: int, directory_name: str = None):
        result: List[Flame] = []
        for frame in range(total_frames):
            clone = deepcopy(self)
            clone.element.attrib["time"] = str(frame + 1)
            # if self.draft:
            #     clone.element.attrib["scale"] = "100"
            # if "rotation" in self.animations:
            #     n_rotations = self.animations["rotation"]["n_rotations"]
            #     new_value = 360 * n_rotations * frame / total_frames % 360
            #     self.element.attrib["rotate"] = str(round(new_value, 4))
            # if "palette" in self.animations:
            #     n_rotations = self.animations["palette"]["n_rotations"]
            #     clone.palette.animate(n_rotations, frame, total_frames)
            for xform in clone.xforms:
                xform.animate(frame)
            result.append(clone)
        time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        dir_name = (
            directory_name or f'rendered-{self.element.attrib["name"]}-{time}'
        )
        draft = "_draft" if self.draft else ""
        return Video(
            result,
            dir_name,
            video_file_name="_" + self.element.attrib["name"] + draft + ".mp4",
            draft=self.draft,
        )

    def __repr__(self):
        return ET.tostring(self.to_element()).decode()
