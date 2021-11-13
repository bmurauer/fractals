import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from fractals.flame import (
    AnimationValue,
    Transform,
    XForm,
    interpolate_linear,
    interpolate_sinusoidal,
    orbit_transform,
)


def test_translation():
    t = Transform("1 0 0 1 0 0")
    t.translate(2, 3)
    assert t.__repr__() == "1.0 0.0 0.0 1.0 2.0 3.0"


def test_rotation():
    t = Transform("1 0 0 1 2 3")
    t.rotate(math.pi)
    assert t.__repr__() == "-1.0 0.0 -0.0 -1.0 2.0 3.0"


def test_sinusoidal_animation():
    value = 0.0
    offset = 1.0
    n_frames = 100
    expected_values = [0.0] * 100
    a = AnimationValue(value, offset)
    for i in range(n_frames):
        v = interpolate_sinusoidal(a, frame=i, n_frames=n_frames)
        print(v)
        # assert v == pytest.approx(expected_values[i])


def test_orbit_transform():
    t = Transform("1 0 0 0 1 0")
    a = AnimationValue(t, orbit=1.0).orbit_transform()
    for i in range(10):
        orbit_transform(a, i, 10)
        print(t.coefs)


def test_linear_animation_single_rotation():
    value = 0
    offset = 1.0
    n_frames = 10
    expected_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2]

    a = AnimationValue(value)
    a.offset = offset
    a.n_repetitions = 1
    for i in range(n_frames):
        v = interpolate_linear(a, frame=i, n_frames=n_frames)
        assert v == pytest.approx(expected_values[i])


def test_linear_animation_multiple_rotations():
    value = 0
    offset = 1.0
    n_frames = 8
    expected_values = [0, 0.5, 1.0, 0.5, 0, 0.5, 1.0, 0.5]

    a = AnimationValue(value)
    a.offset = offset
    a.n_repetitions = 2
    for i in range(n_frames):
        v = interpolate_linear(a, frame=i, n_frames=n_frames)
        assert v == pytest.approx(expected_values[i])


def test_linear_animation_Transform():
    value = Transform("1 0 0 0 1 0")
    offset = Transform("1 0 0 0 1 0")
    n_frames = 10
    expected_values = [
        [1.0, 0, 0, 0, 1.0, 0],
        [1.2, 0, 0, 0, 1.2, 0],
        [1.4, 0, 0, 0, 1.4, 0],
        [1.6, 0, 0, 0, 1.6, 0],
        [1.8, 0, 0, 0, 1.8, 0],
        [2.0, 0, 0, 0, 2.0, 0],
        [1.8, 0, 0, 0, 1.8, 0],
        [1.6, 0, 0, 0, 1.6, 0],
        [1.4, 0, 0, 0, 1.4, 0],
        [1.2, 0, 0, 0, 1.2, 0],
    ]
    a = AnimationValue(value)
    a.offset = offset
    for i in range(n_frames):
        v = (
            interpolate_linear(
                a,
                frame=i,
                n_frames=n_frames,
            )
            .coefs[0:2, :]
            .flatten()
        )
        np.testing.assert_array_almost_equal(v, expected_values[i])


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
