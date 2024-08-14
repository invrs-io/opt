"""Defines filter-and-project density parameterization.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses

import jax.numpy as jnp
from jax import tree_util
from totypes import json_utils, types

from invrs_opt.parameterization import base, transforms


@dataclasses.dataclass
class FilterAndProjectParams(base.ParameterizedDensity2DArrayBase):
    """Stores the latent parameters of the pixel parameterization.

    Attributes:
        latent_density: The latent variable from which the density is obtained.
        beta: Determines the sharpness of the thresholding operation.
    """

    latent_density: types.Density2DArray
    beta: float


tree_util.register_dataclass(
    FilterAndProjectParams,
    data_fields=["latent_density"],
    meta_fields=["beta"],
)

json_utils.register_custom_type(FilterAndProjectParams)


def filter_project(beta: float) -> base.Density2DParameterization:
    """Defines a filter-project parameterization for density arrays.

    The `DensityArray2D` is represented as latent density array that is transformed by,

        transformed = tanh(beta * conv(density.array, gaussian_kernel)) / tanh(beta)

    where the kernel has a full-width at half-maximum determined by the minimum width
    and spacing parameters of the `DensityArray2D`.

    When the density lower and upper bounds are -1 and +1, this basic expression is
    Where the bounds differ, the density is scaled before the transform is applied, and
    then unscaled afterwards.

    Args:
        beta: Determines the sharpness of the thresholding operation.

    Returns:
        The `Density2DParameterization`.
    """

    def from_density_fn(density: types.Density2DArray) -> FilterAndProjectParams:
        """Return latent parameters for the given `density`."""
        array = transforms.normalized_array_from_density(density)
        array = jnp.clip(array, -1, 1)
        array *= jnp.tanh(beta)
        latent_array = jnp.arctanh(array) / beta
        latent_array = transforms.rescale_array_for_density(latent_array, density)
        return FilterAndProjectParams(
            latent_density=dataclasses.replace(density, array=latent_array),
            beta=beta,
        )

    def to_density_fn(params: FilterAndProjectParams) -> types.Density2DArray:
        """Return a density from the latent parameters."""
        transformed = types.symmetrize_density(params.latent_density)
        transformed = transforms.density_gaussian_filter_and_tanh(
            transformed, beta=params.beta
        )
        # Scale to ensure that the full valid range of the density array is reachable.
        mid_value = (transformed.lower_bound + transformed.upper_bound) / 2
        transformed = tree_util.tree_map(
            lambda array: mid_value + (array - mid_value) / jnp.tanh(beta), transformed
        )
        return transforms.apply_fixed_pixels(transformed)

    def constraints_fn(params: FilterAndProjectParams) -> jnp.ndarray:
        """Computes constraints associated with the params."""
        del params
        return jnp.asarray(0.0)

    def update_fn(params: FilterAndProjectParams, step: int) -> FilterAndProjectParams:
        """Perform updates to `params` required for the given `step`."""
        del step
        return params

    return base.Density2DParameterization(
        to_density=to_density_fn,
        from_density=from_density_fn,
        constraints=constraints_fn,
        update=update_fn,
    )
