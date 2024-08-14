"""Base types for density parameterizations.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
from typing import Any, Optional, Protocol, Sequence, Tuple

import jax.numpy as jnp
import numpy as onp
from jax import tree_util
from totypes import json_utils, types

Array = jnp.ndarray | onp.ndarray[Any, Any]
PyTree = Any


class ParameterizedDensity2DArrayBase:
    """Base class for parameterized density arrays."""

    pass


class FromDensityFn(Protocol):
    """Generate the latent representation of a density array."""

    def __call__(
        self, density: types.Density2DArray
    ) -> ParameterizedDensity2DArrayBase:
        ...


class ToDensityFn(Protocol):
    """Generate a density from its latent representation."""

    def __call__(self, params: PyTree) -> types.Density2DArray:
        ...


class ConstraintsFn(Protocol):
    """Compute constraints for a latent representation of a density array."""

    def __call__(self, params: PyTree) -> jnp.ndarray:
        ...


class UpdateFn(Protocol):
    """Performs the required update of a parameterized density for the given step."""

    def __call__(self, params: PyTree, step: int) -> PyTree:
        ...


@dataclasses.dataclass
class Density2DParameterization:
    """Stores `(from_density, to_density, constraints)` function triple."""

    from_density: FromDensityFn
    to_density: ToDensityFn
    constraints: ConstraintsFn
    update: UpdateFn


@dataclasses.dataclass
class Density2DMetadata:
    """Stores the metadata of a `Density2DArray`."""

    lower_bound: float
    upper_bound: float
    fixed_solid: Optional[Array]
    fixed_void: Optional[Array]
    minimum_width: int
    minimum_spacing: int
    periodic: Sequence[bool]
    symmetries: Sequence[str]

    def __post_init__(self) -> None:
        self.periodic = tuple(self.periodic)
        self.symmetries = tuple(self.symmetries)


def _flatten_density_2d_metadata(
    metadata: Density2DMetadata,
) -> Tuple[
    Tuple[()],
    Tuple[
        float,
        float,
        types.HashableWrapper,
        types.HashableWrapper,
        int,
        int,
        Sequence[bool],
        Sequence[str],
    ],
]:
    """Flattens a `Density2DMetadata` into children and auxilliary data."""
    return (
        (),
        (
            metadata.lower_bound,
            metadata.upper_bound,
            types.HashableWrapper(metadata.fixed_solid),
            types.HashableWrapper(metadata.fixed_void),
            metadata.minimum_width,
            metadata.minimum_spacing,
            metadata.periodic,
            metadata.symmetries,
        ),
    )


def _unflatten_density_2d_metadata(
    aux: Tuple[
        float,
        float,
        types.HashableWrapper,
        types.HashableWrapper,
        int,
        int,
        Sequence[bool],
        Sequence[str],
    ],
    children: Tuple[()],
) -> Density2DMetadata:
    """Unflattens a flattened `Density2DMetadata`."""
    del children
    (
        lower_bound,
        upper_bound,
        wrapped_fixed_solid,
        wrapped_fixed_void,
        minimum_width,
        minimum_spacing,
        periodic,
        symmetries,
    ) = aux
    return Density2DMetadata(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        fixed_solid=wrapped_fixed_solid.array,  # type: ignore[arg-type]
        fixed_void=wrapped_fixed_void.array,  # type: ignore[arg-type]
        minimum_width=minimum_width,
        minimum_spacing=minimum_spacing,
        periodic=tuple(periodic),
        symmetries=tuple(symmetries),
    )


tree_util.register_pytree_node(
    Density2DMetadata,
    flatten_func=_flatten_density_2d_metadata,
    unflatten_func=_unflatten_density_2d_metadata,
)

json_utils.register_custom_type(Density2DMetadata)
