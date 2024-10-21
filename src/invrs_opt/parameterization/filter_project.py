"""Defines filter-and-project density parameterization.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses

import jax.numpy as jnp
from jax import tree_util
from totypes import json_utils, types

from invrs_opt.parameterization import base, transforms


@dataclasses.dataclass
class FilterProjectParams(base.ParameterizedDensity2DArray):
    """Stores parameters for the filter-project parameterization."""

    latents: "FilterProjectLatents"
    metadata: "FilterProjectMetadata"


@dataclasses.dataclass
class FilterProjectLatents(base.LatentsBase):
    """Stores latent parameters for the filter-project parameterization.

    Attributes:s
        latent_density: The latent variable from which the density is obtained.
    """

    latent_density: types.Density2DArray


@dataclasses.dataclass
class FilterProjectMetadata(base.MetadataBase):
    """Stores metadata for the filter-project parameterization.

    Attributes:
        beta: Determines the sharpness of the thresholding operation.
    """

    beta: float


tree_util.register_dataclass(
    FilterProjectParams,
    data_fields=["latents", "metadata"],
    meta_fields=[],
)
tree_util.register_dataclass(
    FilterProjectLatents,
    data_fields=["latent_density"],
    meta_fields=[],
)
tree_util.register_dataclass(
    FilterProjectMetadata,
    data_fields=[],
    meta_fields=["beta"],
)
json_utils.register_custom_type(FilterProjectParams)
json_utils.register_custom_type(FilterProjectLatents)
json_utils.register_custom_type(FilterProjectMetadata)


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

    def from_density_fn(density: types.Density2DArray) -> FilterProjectParams:
        """Return latent parameters for the given `density`."""
        density.array = jnp.clip(
            density.array, min=density.lower_bound, max=density.upper_bound
        )
        array = transforms.normalized_array_from_density(density)
        array = jnp.clip(array, -1, 1)
        array *= jnp.tanh(beta)
        latent_array = jnp.arctanh(array) / beta
        latent_array = transforms.rescale_array_for_density(latent_array, density)
        latent_density = density = dataclasses.replace(density, array=latent_array)
        return FilterProjectParams(
            latents=FilterProjectLatents(latent_density=latent_density),
            metadata=FilterProjectMetadata(beta=beta),
        )

    def to_density_fn(params: FilterProjectParams) -> types.Density2DArray:
        """Return a density from the latent parameters."""
        latent_density = params.latents.latent_density
        beta = params.metadata.beta

        transformed = types.symmetrize_density(latent_density)
        transformed = transforms.density_gaussian_filter_and_tanh(transformed, beta)
        # Scale to ensure that the full valid range of the density array is reachable.
        mid_value = (transformed.lower_bound + transformed.upper_bound) / 2
        transformed = tree_util.tree_map(
            lambda array: mid_value + (array - mid_value) / jnp.tanh(beta), transformed
        )
        return transforms.apply_fixed_pixels(transformed)

    def constraints_fn(params: FilterProjectParams) -> jnp.ndarray:
        """Computes constraints associated with the params."""
        del params
        return jnp.asarray(0.0)

    def update_fn(
        params: FilterProjectParams,
        updates: FilterProjectParams,
        value: jnp.ndarray,
        step: int,
    ) -> FilterProjectParams:
        """Perform updates to `params` required for the given `step`."""
        del step, value
        return FilterProjectParams(
            latents=tree_util.tree_map(
                lambda a, b: a + b, params.latents, updates.latents
            ),
            metadata=params.metadata,
        )

    return base.Density2DParameterization(
        to_density=to_density_fn,
        from_density=from_density_fn,
        constraints=constraints_fn,
        update=update_fn,
    )
