import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from fractals.flame import (
    Transform,
    XForm, orbit, Flame,
)
from fractals.utils import get_flame_from_file


def test_translation():
    t = Transform("1 0 0 1 0 0")
    t.translate(2, 3)
    np.testing.assert_array_almost_equal(
        np.array([[1, 0, 2], [0, 1, 3], [1, 1, 1]]),
        t.coefs
    )


def test_rotation():
    t = Transform("1 0 0 1 2 3")
    t.rotate(math.pi)
    np.testing.assert_array_almost_equal(
        np.array([[-1, 0, 2], [0, -1, 3], [1, 1, 1]]),
        t.coefs
    )


def test_orbit():
    f = Flame.from_element(get_flame_from_file("heartgrid.flame"))
    f.xforms[0].add_orbit_animation(radius=3.0)
    expected_coefs =[
        [[1, 0, 0], [0, 1, 0], [1, 1, 1]],
        [[1, 0, -3], [0, 1, 3], [1, 1, 1]],
        [[1, 0, -6], [0, 1, 0], [1, 1, 1]],
        [[1, 0, -3], [0, 1, -3], [1, 1, 1]],
    ]
    n_frames = 4
    flames = f.animate(n_frames)
    for i in range(n_frames):
        np.testing.assert_array_almost_equal(
            np.array(expected_coefs[i]),
            flames.flames[i].xforms[0].coefs.coefs
        )


def test_linear_animation_Transform():
    f = Flame.from_element(get_flame_from_file("heartgrid.flame"))
    f.xforms[0].coefs = Transform("1 0 0 1 1 0")
    offset = Transform("0 0 0 0 1 0")
    offset.coefs = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
    ])
    f.xforms[0].add_translation_animation(offset)

    n_frames = 10
    expected_coefs = [
        [[1.0, 0, 1.0], [0, 1, 0], [1, 1, 1]],
        [[1.0, 0, 1.2], [0, 1, 0], [1, 1, 1]],
        [[1.0, 0, 1.4], [0, 1, 0], [1, 1, 1]],
        [[1.0, 0, 1.6], [0, 1, 0], [1, 1, 1]],
        [[1.0, 0, 1.8], [0, 1, 0], [1, 1, 1]],
        [[1.0, 0, 2.0], [0, 1, 0], [1, 1, 1]],
        [[1.0, 0, 1.8], [0, 1, 0], [1, 1, 1]],
        [[1.0, 0, 1.6], [0, 1, 0], [1, 1, 1]],
        [[1.0, 0, 1.4], [0, 1, 0], [1, 1, 1]],
        [[1.0, 0, 1.2], [0, 1, 0], [1, 1, 1]],
    ]

    flames = f.animate(n_frames)
    for i in range(n_frames):
        np.testing.assert_array_almost_equal(
            np.array(expected_coefs[i]),
            flames.flames[i].xforms[0].coefs.coefs
        )



def test_xform_color_animation():
    xform = XForm(
        element=ET.Element("xform"),
        coefs=AnimationValue(Transform("1 0 0 0 1 0")),
        color=AnimationValue(0.0).interpolate_linear(1.0),
    )

    xform.animate(5, 10)
    assert xform.color.value == pytest.approx(1.0)

    # should be idempotent
    xform.animate(5, 10)
    assert xform.color.value == pytest.approx(1.0)


def test_xform_coef_animation():
    xform = XForm(
        element=ET.Element("xform"),
        coefs=AnimationValue(Transform("1 0 0 0 1 0")).interpolate_linear(
            Transform("1 0 0 0 1 0"),
            n_rotations=1,
        ),
        color=AnimationValue(0.0),
    )

    xform.animate(2, 10)
    np.testing.assert_array_almost_equal(
        xform.coefs.value.coefs[0:2, :].flatten(), np.array([1.4, 0, 0, 0, 1.4, 0])
    )

    xform.animate(5, 10)
    assert xform.coefs.__repr__() == "2.0 0.0 0.0 0.0 2.0 0.0"
