import logging
import random
import xml.etree.ElementTree as ET
from typing import Optional

logger = logging.getLogger("Logger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y%m%d-%H:%M:%S"
)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)


def get_flame_from_file(
    file_name: str, flame_name: Optional[str] = None, flame_idx: Optional[int] = None
) -> ET.Element:
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
