"""Tests for `transforms`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized
from totypes import types

from invrs_opt.parameterization import transforms


class GaussianFilterTest(unittest.TestCase):
    @parameterized.expand([[1, 5], [3, 3], [5, 1]])
    def test_transformed_matches_expected(self, minimum_width, minimum_spacing):
        array = onp.zeros((9, 9))
        array[4, 4] = 9
        density = types.Density2DArray(
            array=array,
            lower_bound=0,
            upper_bound=1,
            minimum_width=minimum_width,
            minimum_spacing=minimum_spacing,
        )
        beta = 1
        transformed = transforms.density_gaussian_filter_and_tanh(density, beta=beta)
        expected = onp.asarray(
            [
                [0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
                [0.12, 0.12, 0.13, 0.14, 0.14, 0.14, 0.13, 0.12, 0.12],
                [0.12, 0.13, 0.15, 0.22, 0.27, 0.22, 0.15, 0.13, 0.12],
                [0.12, 0.14, 0.22, 0.48, 0.64, 0.48, 0.22, 0.14, 0.12],
                [0.12, 0.14, 0.27, 0.64, 0.82, 0.64, 0.27, 0.14, 0.12],
                [0.12, 0.14, 0.22, 0.48, 0.64, 0.48, 0.22, 0.14, 0.12],
                [0.12, 0.13, 0.15, 0.22, 0.27, 0.22, 0.15, 0.13, 0.12],
                [0.12, 0.12, 0.13, 0.14, 0.14, 0.14, 0.13, 0.12, 0.12],
                [0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
            ]
        )
        onp.testing.assert_allclose(transformed.array, expected, rtol=0.05)

    @parameterized.expand([[1, 1], [3, 1], [5, 1], [10, 1], [10, 0.5], [10, 2]])
    def test_ones_density_yields_tanh_beta(self, length_scale, upper_bound):
        array = onp.ones((20, 20)) * upper_bound
        density = types.Density2DArray(
            array=array,
            lower_bound=0,
            upper_bound=upper_bound,
            minimum_width=length_scale,
            minimum_spacing=length_scale,
        )
        beta = 1
        transformed = transforms.density_gaussian_filter_and_tanh(density, beta=beta)
        onp.testing.assert_allclose(
            transformed.array,
            (1 + onp.tanh(beta)) * 0.5 * upper_bound,
            rtol=0.01,
        )

    def test_batch_matches_single(self):
        beta = 4
        density = types.Density2DArray(
            array=onp.arange(600).reshape((6, 10, 10)),
            minimum_width=5,
            minimum_spacing=5,
        )
        transformed = transforms.density_gaussian_filter_and_tanh(density, beta=beta)
        for i in range(6):
            transformed_single = transforms.density_gaussian_filter_and_tanh(
                density=dataclasses.replace(
                    density,
                    array=density.array[i, :, :],
                ),
                beta=beta,
            )
            onp.testing.assert_allclose(
                transformed.array[i, :, :], transformed_single.array
            )

    def test_periodic(self):
        beta = 100
        array = onp.zeros((5, 5))
        array[0, 0] = 9

        # No periodicity.
        density = types.Density2DArray(
            array,
            minimum_spacing=3,
            minimum_width=3,
            periodic=(False, False),
            lower_bound=0,
            upper_bound=1,
        )
        transformed = transforms.density_gaussian_filter_and_tanh(density, beta=beta)
        expected = onp.asarray(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        onp.testing.assert_allclose(transformed.array, expected, atol=0.01)

        # Periodic along the first axis.
        density = dataclasses.replace(density, periodic=(True, False))
        transformed = transforms.density_gaussian_filter_and_tanh(density, beta=beta)
        expected = onp.asarray(
            [
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
            ]
        )
        onp.testing.assert_allclose(transformed.array, expected, atol=0.01)

        # Periodic along the second axis.
        density = dataclasses.replace(density, periodic=(False, True))
        transformed = transforms.density_gaussian_filter_and_tanh(density, beta=beta)
        expected = onp.asarray(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        onp.testing.assert_allclose(transformed.array, expected, atol=0.01)

        # Periodic along both axes.
        density = dataclasses.replace(density, periodic=(True, True))
        transformed = transforms.density_gaussian_filter_and_tanh(density, beta=beta)
        expected = onp.asarray(
            [
                [1, 1, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ]
        )
        onp.testing.assert_allclose(transformed.array, expected, atol=0.01)


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
        array = transforms.normalized_array_from_density(density)
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
        rescaled = transforms.rescale_array_for_density(array, dummy_density)
        expected = jnp.linspace(lower_bound, upper_bound).reshape((10, 5))
        onp.testing.assert_allclose(rescaled, expected, rtol=1e-5)


class FixedPixelTest(unittest.TestCase):
    @parameterized.expand(
        [
            [
                onp.asarray([[1, 0, 0, 0, 0]], dtype=bool),
                onp.asarray([[0, 0, 1, 1, 0]], dtype=bool),
                onp.asarray([[3.0, 0.0, -0.5, -0.5, 0.0]]),
            ],
            [
                None,
                onp.asarray([[0, 0, 1, 1, 0]], dtype=bool),
                onp.asarray([[0.0, 0.0, -0.5, -0.5, 0.0]]),
            ],
            [
                None,
                None,
                onp.asarray([[0.0, 0.0, 0.0, 0.0, 0.0]]),
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
        density = transforms.apply_fixed_pixels(density)
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
        padded = transforms.pad2d(array, ((1, 1), (1, 1)), "edge")
        padded_both_specified = transforms.pad2d(
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
        padded = transforms.pad2d(array, ((1, 1), (1, 1)), "wrap")
        padded_both_specified = transforms.pad2d(
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
        padded = transforms.pad2d(array, ((1, 1), (1, 1)), ("wrap", "edge"))
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
        padded = transforms.pad2d(array, ((1, 1), (1, 1)), ("edge", "wrap"))
        onp.testing.assert_array_equal(padded, expected)

    def test_pad2d_batch_dims(self):
        array = jnp.arange(300).reshape((2, 1, 5, 10, 3))
        pad_width = ((3, 4), (1, 2))
        padded = transforms.pad2d(array, pad_width, "wrap")
        self.assertSequenceEqual(padded.shape, (2, 1, 5, 17, 6))
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                for k in range(array.shape[2]):
                    onp.testing.assert_array_equal(
                        padded[i, j, k, :, :],
                        transforms.pad2d(array[i, j, k, :, :], pad_width, "wrap"),
                    )


class PadWidthForKernelShapeTest(unittest.TestCase):
    @parameterized.expand([[(2, 3)], [(4, 4)], [(5, 6)]])
    def test_pad_width(self, kernel_shape):
        # Checks that when padded by the computed amount, a convolution
        # valid padding returns an array with the original shape.
        kernel = jnp.ones(kernel_shape)
        array = jnp.arange(77).reshape((7, 11)).astype(float)
        pad_width = transforms.pad_width_for_kernel_shape(kernel_shape)
        padded = jnp.pad(array, pad_width)
        y = jax.lax.conv_general_dilated(
            lhs=padded[jnp.newaxis, jnp.newaxis, :, :],  # HCHW
            rhs=kernel[jnp.newaxis, jnp.newaxis, :, :],  # OIHW
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            window_strides=(1, 1),
        )
        self.assertEqual(y.shape, (1, 1) + array.shape)

    @parameterized.expand([[(2, 3)], [(4, 4)], [(5, 6)]])
    def test_pad_width_offsets(self, kernel_shape):
        kernel = onp.zeros(kernel_shape)
        kernel[kernel_shape[0] // 2, kernel_shape[1] // 2] = 1
        array = jnp.arange(77).reshape((7, 11)).astype(float)
        pad_width = transforms.pad_width_for_kernel_shape(kernel_shape)
        padded = jnp.pad(array, pad_width)
        y = jax.lax.conv_general_dilated(
            lhs=padded[jnp.newaxis, jnp.newaxis, :, :],  # HCHW
            rhs=kernel[jnp.newaxis, jnp.newaxis, :, :],  # OIHW
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            window_strides=(1, 1),
        )
        onp.testing.assert_array_equal(y[0, 0, ...], array)


class GaussianKernelTest(unittest.TestCase):
    @parameterized.expand([[2], [3], [4], [5]])
    def test_gaussian_peak_on_gridpoint(self, fwhm_size_multiple):
        kernel = transforms._gaussian_kernel(1.0, fwhm_size_multiple)
        self.assertEqual(kernel[kernel.shape[0] // 2, kernel.shape[1] // 2], 1.0)


class InterfacePixelsTest(unittest.TestCase):
    def test_interface_pixels_match_expected(self):
        array = -0.5 + jnp.asarray(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        expected = jnp.asarray(
            [
                [0, 0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1],
            ],
            dtype=bool,
        )
        onp.testing.assert_array_equal(
            transforms.interface_pixels(array, periodic=(False, False)), expected
        )

    def test_interface_pixels_match_expected_periodic(self):
        array = -0.5 + jnp.asarray(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        expected = jnp.asarray(
            [
                [0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1],
            ],
            dtype=bool,
        )
        onp.testing.assert_array_equal(
            transforms.interface_pixels(array, periodic=(True, True)), expected
        )
