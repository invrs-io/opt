"""Defines transforms for custom types and bare jax arrays.

Copyright (c) 2023 The INVRS-IO authors.
"""

from typing import Tuple, Union

import jax
import jax.numpy as jnp
from jax import tree_util
from totypes import types

PadMode = Union[str, Tuple[str, str]]

GAUSSIAN_FWHM_SIZE_MULTIPLE: float = 3.0


def density_gaussian_filter_and_tanh(
    density: types.Density2DArray,
    beta: float,
    fwhm_size_multiple: float = GAUSSIAN_FWHM_SIZE_MULTIPLE,
) -> types.Density2DArray:
    """Filters the density using a Gaussian kernel and applies a tanh nonlinearity.

    The Gaussian full width at half maximum is the average of the minimum width and
    minimum spacing for the density.

    Args:
        density: The density to be transformed.
        beta: Scalar determining the strength of the tanh nonlinearity.
        fwhm_size_multiple: Determines the size of the Gaussian kernel; if the full
            width is `fwhm`, then the kernel size is `ceil(fwhm_size_multiple * fwhm)`.

    Returns:
        The transformed `density`.
    """
    array = normalized_array_from_density(density)
    pad_mode = _pad_mode_for_density(density)

    batch_shape = array.shape[:-2]
    array_flat_batch = array.reshape((-1, 1) + array.shape[-2:])

    kernel = _gaussian_kernel(
        (density.minimum_width + density.minimum_spacing) / 2,
        fwhm_size_multiple=fwhm_size_multiple,
    )
    kernel /= jnp.sum(kernel)
    kernel = kernel[jnp.newaxis, jnp.newaxis, :, :]  # OIHW format

    pad_width = pad_width_for_kernel_shape(kernel.shape)
    transformed = pad2d(array_flat_batch, pad_width, pad_mode=pad_mode)
    transformed = conv(transformed, kernel, padding="VALID")
    transformed = jnp.tanh(beta * transformed)
    transformed = transformed[..., 0, :, :]  # Remove the dummy channel axis.
    transformed = transformed.reshape((batch_shape) + array.shape[-2:])
    transformed = rescale_array_for_density(transformed, density)

    treedef = jax.tree_util.tree_structure(density)
    transformed_density: types.Density2DArray = jax.tree_util.tree_unflatten(
        treedef, (transformed,)
    )
    return transformed_density


def normalized_array_from_density(density: types.Density2DArray) -> jnp.ndarray:
    """Returns an array with values scaled to the range `(-1, 1)`."""
    value_mid = (density.upper_bound + density.lower_bound) / 2
    value_range = density.upper_bound - density.lower_bound
    return jnp.asarray(2 * (density.array - value_mid) / value_range)


def rescale_array_for_density(
    array: jnp.ndarray,
    density: types.Density2DArray,
) -> jnp.ndarray:
    """Rescales an array for the bounds defined by `density`."""
    value_mid = (density.upper_bound + density.lower_bound) / 2
    value_range = density.upper_bound - density.lower_bound
    return array / 2 * value_range + value_mid


def apply_fixed_pixels(density: types.Density2DArray) -> types.Density2DArray:
    """Set fixed pixels with their required values."""
    fixed_solid = density.fixed_solid
    fixed_void = density.fixed_void
    (array,), treedef = tree_util.tree_flatten(density)
    if fixed_solid is not None:
        array = jnp.where(fixed_solid, density.upper_bound, array)
    if fixed_void is not None:
        array = jnp.where(fixed_void, density.lower_bound, array)
    transformed_density: types.Density2DArray = tree_util.tree_unflatten(
        treedef, (array,)
    )
    return transformed_density


def conv(x: jnp.ndarray, kernel: jnp.ndarray, padding: str) -> jnp.ndarray:
    """Convolves `x` with `kernel`."""
    assert x.ndim == 4
    assert kernel.ndim == 4
    return jax.lax.conv_general_dilated(
        lhs=x,
        rhs=kernel,
        padding=padding,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        window_strides=(1, 1),
    )


def pad_width_for_kernel_shape(
    kernel_shape: Tuple[int, ...],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return the pad width for the trailing dimensions of the kernel."""
    kernel_height, kernel_width = kernel_shape[-2:]
    return (
        (kernel_height // 2, kernel_height // 2 - (kernel_height + 1) % 2),
        (kernel_width // 2, kernel_width // 2 - (kernel_width + 1) % 2),
    )


def pad2d(
    x: jnp.ndarray,
    pad_width: Tuple[Tuple[int, int], Tuple[int, int]],
    pad_mode: PadMode,
) -> jnp.ndarray:
    """Pad the trailing two axes of `x` as specified."""
    leading_pad_width = ((0, 0),) * (x.ndim - 2)
    if isinstance(pad_mode, str):
        return jnp.pad(x, leading_pad_width + pad_width, mode=pad_mode)

    pad_width_i, pad_width_j = pad_width
    pad_mode_i, pad_mode_j = pad_mode
    x_padded_i = jnp.pad(
        x,
        pad_width=leading_pad_width + (pad_width_i, (0, 0)),
        mode=pad_mode_i,
    )
    return jnp.pad(
        x_padded_i,
        pad_width=leading_pad_width + ((0, 0), pad_width_j),
        mode=pad_mode_j,
    )


def _pad_mode_for_density(density: types.Density2DArray) -> Union[str, Tuple[str, str]]:
    """Return the pad mode implied by the `periodic` attribute of `density`."""
    if density.periodic == (True, True):
        return "wrap"
    if density.periodic == (False, False):
        return "edge"
    return (
        "wrap" if density.periodic[0] else "edge",
        "wrap" if density.periodic[1] else "edge",
    )


def _gaussian_kernel(fwhm: float, fwhm_size_multiple: float) -> jnp.ndarray:
    """Returns a Gaussian kernel with the specified full-width at half-maximum."""
    kernel_size = max(1, int(jnp.ceil(fwhm * fwhm_size_multiple)))
    # Ensure the kernel size is odd, so that there is always a central pixel which will
    # contain the peak value of the Gaussian.
    kernel_size += (kernel_size + 1) % 2
    d = jnp.arange(0.5, kernel_size) - kernel_size / 2
    x = d[:, jnp.newaxis]
    y = d[jnp.newaxis, :]
    sigma = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
    return jnp.exp(-(x**2 + y**2) / (2 * sigma**2))
