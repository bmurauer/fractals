import unittest

import xml.etree.ElementTree as ET
from flame import Color, rotate_colors, Flame, Palette, XForm, Transform


def create_test_palette() -> Palette:
    return Palette(
        count=2,
        colors=[
            Color(0, 0, 128),
            Color(64, 0, 0),
        ]
    )


def create_test_xform() -> XForm:
    xform = XForm(
        element=ET.Element('xform'),
        coefs=Transform("1 0 0 0 1 0"),
        color=0.0,
    )
    xform.element.attrib['linear'] = str(1.0)
    return xform


def create_test_flame() -> Flame:
    
    return Flame(
        element=ET.Element('flame'),
        xforms=[create_test_xform(), create_test_xform()],
        palette=create_test_palette(),
    )


class FlameTests(unittest.TestCase):

    def test_color_interpolation(self):
        c1 = Color(0, 0, 254)
        c2 = Color(128, 0, 0)
        c3 = c1.interpolate_towards(c2, 0.5)
        self.assertEqual(c3.r, 64)
        self.assertEqual(c3.g, 0)
        self.assertEqual(c3.b, 127)

    def test_frame_color_rotation(self):
        flame = create_test_flame()
        flame.palette = Palette(count=4, colors=[
            Color.from_hex("0000f0"),  # f0 = 240
            Color.from_hex("00f000"),
            Color.from_hex("f00000"),
            Color.from_hex("000000"),
        ])
            
        flames = rotate_colors(flame, 8)

        self.assertEqual("0000f0", str(flames[0].palette.colors[0]))
        self.assertEqual("00f000", str(flames[0].palette.colors[1]))
        self.assertEqual("f00000", str(flames[0].palette.colors[2]))
        self.assertEqual("000000", str(flames[0].palette.colors[3]))

        self.assertEqual("007878", str(flames[1].palette.colors[0]))  # 78 = 120
        self.assertEqual("787800", str(flames[1].palette.colors[1]))
        self.assertEqual("780000", str(flames[1].palette.colors[2]))
        self.assertEqual("000078", str(flames[1].palette.colors[3]))

        self.assertEqual("00f000", str(flames[2].palette.colors[0]))  # one "complete"
        self.assertEqual("f00000", str(flames[2].palette.colors[1]))  # color was cycled
        self.assertEqual("000000", str(flames[2].palette.colors[2]))
        self.assertEqual("0000f0", str(flames[2].palette.colors[3]))
