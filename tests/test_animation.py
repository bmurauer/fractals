import math

import numpy as np

from fractals.flame import Flame, Transform
from fractals.utils import get_flame_from_file


def test_translation():
    t = Transform("1 0 0 1 0 0")
    t.translate(2, 3)
    np.testing.assert_array_almost_equal(
        np.array([[1, 0, 2], [0, 1, 3], [1, 1, 1]]), t.coefs
    )


def test_rotation():
    t = Transform("1 0 0 1 2 3")
    t.rotate(math.pi)
    np.testing.assert_array_almost_equal(
        np.array([[-1, 0, 2], [0, -1, 3], [1, 1, 1]]), t.coefs
    )


def test_orbit():
    f = Flame.from_element(get_flame_from_file("heartgrid.flame"))
    f.xforms[0].add_orbit_animation(radius=3.0)
    expected_coefs = [
        [[1, 0, 0], [0, 1, 0], [1, 1, 1]],
        [[1, 0, -3], [0, 1, 3], [1, 1, 1]],
        [[1, 0, -6], [0, 1, 0], [1, 1, 1]],
        [[1, 0, -3], [0, 1, -3], [1, 1, 1]],
    ]
    n_frames = 4
    flames = f.animate(n_frames)
    for i in range(n_frames):
        np.testing.assert_array_almost_equal(
            np.array(expected_coefs[i]), flames.flames[i].xforms[0].coefs.coefs
        )


def test_linear_animation_transform():
    f = Flame.from_element(get_flame_from_file("heartgrid.flame"))
    f.xforms[0].fransform = Transform("1 0 0 1 1 0")
    offset = Transform("0 0 0 0 1 0")
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
            np.array(expected_coefs[i]), flames.flames[i].xforms[0].coefs.coefs
        )
