"""Defines the direct pixel parameterization for density arrays.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses

import jax.numpy as jnp
from totypes import types

from invrs_opt.parameterization import base


@dataclasses.dataclass
class PixelParams(base.ParameterizedDensity2DArrayBase):
    """Stores latent parameters of the direct pixel parameterization."""

    density: types.Density2DArray


def pixel() -> base.Density2DParameterization:
    """Return the direct pixel parameterization."""

    def from_density_fn(density: types.Density2DArray) -> PixelParams:
        return PixelParams(density=density)

    def to_density_fn(params: PixelParams) -> types.Density2DArray:
        return params.density

    def constraints_fn(params: PixelParams) -> jnp.ndarray:
        del params
        return jnp.asarray(0.0)

    return base.Density2DParameterization(
        from_density=from_density_fn,
        to_density=to_density_fn,
        constraints=constraints_fn,
    )
