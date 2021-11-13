import unittest
import xml.etree.ElementTree as ET

from fractals.flame import (AnimationValue, Color, Flame, Palette, Transform,
                            XForm)


def create_test_palette() -> Palette:
    return Palette(
        count=2,
        colors=[
            Color(0, 0, 128),
            Color(64, 0, 0),
        ],
    )


def create_test_xform() -> XForm:
    xform = XForm(
        element=ET.Element("xform"),
        coefs=AnimationValue(Transform("1 0 0 0 1 0")),
        color=AnimationValue(0.0),
    )
    xform.element.attrib["linear"] = str(1.0)
    return xform


def create_test_flame() -> Flame:

    return Flame(
        element=ET.Element("flame"),
        xforms=[create_test_xform(), create_test_xform()],
        palette=create_test_palette(),
    )


def test_color_interpolation():
    c1 = Color(0, 0, 254)
    c2 = Color(128, 0, 0)
    c3 = c1.interpolate_towards(c2, 0.5)
    assert c3.r == 64
    assert c3.g == 0
    assert c3.b == 127


def test_frame_color_rotation():
    flame = create_test_flame()
    flame.element.attrib["name"] = "blob"
    flame.palette = Palette(
        count=4,
        colors=[
            Color.from_hex("0000f0"),  # f0 = 240
            Color.from_hex("00f000"),
            Color.from_hex("f00000"),
            Color.from_hex("000000"),
        ],
    )

    flames = flame.rotate_colors(8).flames

    assert "0000f0" == str(flames[0].palette.colors[0])
    assert "00f000" == str(flames[0].palette.colors[1])
    assert "f00000" == str(flames[0].palette.colors[2])
    assert "000000" == str(flames[0].palette.colors[3])

    assert "007878" == str(flames[1].palette.colors[0])  # 78 = 120
    assert "787800" == str(flames[1].palette.colors[1])
    assert "780000" == str(flames[1].palette.colors[2])
    assert "000078" == str(flames[1].palette.colors[3])

    assert "00f000" == str(flames[2].palette.colors[0])  # one "complete"
    assert "f00000" == str(flames[2].palette.colors[1])  # color was cycled
    assert "000000" == str(flames[2].palette.colors[2])
    assert "0000f0" == str(flames[2].palette.colors[3])


def test_frame_color_rotation_2():
    flame = create_test_flame()
    flame.element.attrib["name"] = "blob"
    flame.palette = Palette(
        count=2,
        colors=[
            Color.from_hex("000010"),  # 10hex = 16dec
            Color.from_hex("000000"),
        ],
    )

    flames = flame.rotate_colors(16).flames
    # in the "flames" list are now 16 flames that cover a whole rotation of the palette.

    assert str(flames[0].palette.colors[0]) == "000010"
    assert str(flames[0].palette.colors[1]) == "000000"

    # one of 8 steps from 16 towards 0 should be 14
    assert str(flames[1].palette.colors[0]) == "00000e"
    assert str(flames[1].palette.colors[1]) == "000002"

    assert str(flames[2].palette.colors[0]) == "00000c"
    assert str(flames[2].palette.colors[1]) == "000004"

    assert str(flames[3].palette.colors[0]) == "00000a"
    assert str(flames[3].palette.colors[1]) == "000006"

    assert str(flames[4].palette.colors[0]) == "000008"
    assert str(flames[4].palette.colors[1]) == "000008"

    # one of 8 steps from 16 towards 0 should be 14
    assert str(flames[5].palette.colors[0]) == "000006"
    assert str(flames[5].palette.colors[1]) == "00000a"

    assert str(flames[6].palette.colors[0]) == "000004"
    assert str(flames[6].palette.colors[1]) == "00000c"

    assert str(flames[7].palette.colors[0]) == "000002"
    assert str(flames[7].palette.colors[1]) == "00000e"
