"""Tests for `lbfgsb.transforms`.

Copyright (c) 2023 Martin F. Schubert
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

from invrs_opt.lbfgsb import transform
from totypes import types

TEST_KERNEL = onp.asarray(  # Kernel is intentionally asymmetric.
    [
        [0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
    ],
    dtype=bool,
)


class RescaleTest(unittest.TestCase):
    @parameterized.expand([(-1.0, 1.0), (2.0, 3.0), (-0.5, -0.1)])
    def test_normalized_array_from_density(self, lower_bound, upper_bound):
        density = types.Density2DArray(
            array=jnp.linspace(lower_bound, upper_bound).reshape((10, 5)),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            fixed_solid=None,
            fixed_void=None,
            minimum_width=1,
            minimum_spacing=1,
        )
        # Compute `array`, which should now have values between `-1` and `1`.
        array = transform.normalized_array_from_density(density)
        expected = jnp.linspace(-1.0, 1.0).reshape((10, 5))
        onp.testing.assert_allclose(array, expected, rtol=1e-5)

    @parameterized.expand([(-1.0, 1.0), (2.0, 3.0), (-0.5, -0.1)])
    def test_rescale_array_for_density(self, lower_bound, upper_bound):
        dummy_density = types.Density2DArray(
            array=jnp.ones((2, 2)),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            fixed_solid=None,
            fixed_void=None,
            minimum_width=1,
            minimum_spacing=1,
        )
        array = jnp.linspace(-1.0, 1.0).reshape((10, 5))
        rescaled = transform.rescale_array_for_density(array, dummy_density)
        expected = jnp.linspace(lower_bound, upper_bound).reshape((10, 5))
        onp.testing.assert_allclose(rescaled, expected, rtol=1e-5)


class FixedPixelTest(unittest.TestCase):
    @parameterized.expand(
        [
            [
                jnp.asarray([[1, 0, 0, 0, 0]], dtype=bool),
                jnp.asarray([[0, 0, 1, 1, 0]], dtype=bool),
                jnp.asarray([[3.0, 0.0, -0.5, -0.5, 0.0]]),
            ],
            [
                None,
                jnp.asarray([[0, 0, 1, 1, 0]], dtype=bool),
                jnp.asarray([[0.0, 0.0, -0.5, -0.5, 0.0]]),
            ],
            [
                None,
                None,
                jnp.asarray([[0.0, 0.0, 0.0, 0.0, 0.0]]),
            ],
        ]
    )
    def test_apply_fixed_pixels(self, fixed_solid, fixed_void, expected):
        density = types.Density2DArray(
            array=jnp.zeros((1, 5)),
            fixed_solid=fixed_solid,
            fixed_void=fixed_void,
            lower_bound=-0.5,
            upper_bound=3.0,
        )
        density = transform.apply_fixed_pixels(density)
        onp.testing.assert_array_equal(density.array, expected)


class Pad2DTest(unittest.TestCase):
    def test_pad2d_edge(self):
        array = jnp.asarray(
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
            ]
        )
        expected = jnp.asarray(
            [
                [0, 0, 1, 2, 3, 4, 4],
                [0, 0, 1, 2, 3, 4, 4],
                [5, 5, 6, 7, 8, 9, 9],
                [10, 10, 11, 12, 13, 14, 14],
                [10, 10, 11, 12, 13, 14, 14],
            ]
        )
        padded = transform.pad2d(array, ((1, 1), (1, 1)), "edge")
        padded_both_specified = transform.pad2d(
            array, ((1, 1), (1, 1)), ("edge", "edge")
        )
        onp.testing.assert_array_equal(padded, expected)
        onp.testing.assert_array_equal(padded_both_specified, expected)

    def test_pad2d_wrap(self):
        array = jnp.asarray(
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
            ]
        )
        expected = jnp.asarray(
            [
                [14, 10, 11, 12, 13, 14, 10],
                [4, 0, 1, 2, 3, 4, 0],
                [9, 5, 6, 7, 8, 9, 5],
                [14, 10, 11, 12, 13, 14, 10],
                [4, 0, 1, 2, 3, 4, 0],
            ]
        )
        padded = transform.pad2d(array, ((1, 1), (1, 1)), "wrap")
        padded_both_specified = transform.pad2d(
            array, ((1, 1), (1, 1)), ("wrap", "wrap")
        )
        onp.testing.assert_array_equal(padded, expected)
        onp.testing.assert_array_equal(padded_both_specified, expected)

    def test_pad2d_wrap_edge(self):
        array = jnp.asarray(
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
            ]
        )
        expected = jnp.asarray(
            [
                [10, 10, 11, 12, 13, 14, 14],
                [0, 0, 1, 2, 3, 4, 4],
                [5, 5, 6, 7, 8, 9, 9],
                [10, 10, 11, 12, 13, 14, 14],
                [0, 0, 1, 2, 3, 4, 4],
            ]
        )
        padded = transform.pad2d(array, ((1, 1), (1, 1)), ("wrap", "edge"))
        onp.testing.assert_array_equal(padded, expected)

    def test_pad2d_edge_wrap(self):
        array = jnp.asarray(
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
            ]
        )
        expected = jnp.asarray(
            [
                [4, 0, 1, 2, 3, 4, 0],
                [4, 0, 1, 2, 3, 4, 0],
                [9, 5, 6, 7, 8, 9, 5],
                [14, 10, 11, 12, 13, 14, 10],
                [14, 10, 11, 12, 13, 14, 10],
            ]
        )
        padded = transform.pad2d(array, ((1, 1), (1, 1)), ("edge", "wrap"))
        onp.testing.assert_array_equal(padded, expected)

    def test_pad2d_batch_dims(self):
        array = jnp.arange(300).reshape((2, 1, 5, 10, 3))
        pad_width = ((3, 4), (1, 2))
        padded = transform.pad2d(array, pad_width, "wrap")
        self.assertSequenceEqual(padded.shape, (2, 1, 5, 17, 6))
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                for k in range(array.shape[2]):
                    onp.testing.assert_array_equal(
                        padded[i, j, k, :, :],
                        transform.pad2d(array[i, j, k, :, :], pad_width, "wrap"),
                    )


class PadWidthForKernelShapeTest(unittest.TestCase):
    @parameterized.expand([[(2, 3)], [(4, 4)], [(5, 6)]])
    def test_pad_width(self, kernel_shape):
        # Checks that when padded by the computed amount, a convolution
        # valid padding returns an array with the original shape.
        kernel = jnp.ones(kernel_shape)
        array = jnp.arange(77).reshape((7, 11)).astype(float)
        pad_width = transform.pad_width_for_kernel_shape(kernel_shape)
        padded = jnp.pad(array, pad_width)
        y = jax.lax.conv_general_dilated(
            lhs=padded[jnp.newaxis, jnp.newaxis, :, :],  # HCHW
            rhs=kernel[jnp.newaxis, jnp.newaxis, :, :],  # OIHW
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            window_strides=(1, 1),
        )
        self.assertEqual(y.shape, (1, 1) + array.shape)
