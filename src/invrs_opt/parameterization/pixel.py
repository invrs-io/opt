"""Defines the direct pixel parameterization for density arrays.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses

import jax.numpy as jnp
from jax import tree_util
from totypes import json_utils, types

from invrs_opt.parameterization import base


@dataclasses.dataclass
class PixelParams(base.ParameterizedDensity2DArray):
    latents: "PixelLatents"
    metadata: None = None


@dataclasses.dataclass
class PixelLatents(base.LatentsBase):
    """Stores latent parameters for the direct pixel parameterization."""

    density: types.Density2DArray


tree_util.register_dataclass(
    PixelParams,
    data_fields=["latents"],
    meta_fields=[],
)
tree_util.register_dataclass(
    PixelLatents,
    data_fields=["density"],
    meta_fields=[],
)
json_utils.register_custom_type(PixelParams)
json_utils.register_custom_type(PixelLatents)


def pixel() -> base.Density2DParameterization:
    """Return the direct pixel parameterization."""

    def from_density_fn(density: types.Density2DArray) -> PixelParams:
        return PixelParams(latents=PixelLatents(density=density))

    def to_density_fn(params: PixelParams) -> types.Density2DArray:
        return params.latents.density

    def constraints_fn(params: PixelParams) -> jnp.ndarray:
        del params
        return jnp.asarray(0.0)

    def update_fn(params: PixelParams, step: int) -> PixelParams:
        del step
        return params

    return base.Density2DParameterization(
        from_density=from_density_fn,
        to_density=to_density_fn,
        constraints=constraints_fn,
        update=update_fn,
    )
