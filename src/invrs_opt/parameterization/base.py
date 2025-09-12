"""Base types for density parameterizations.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
from typing import Any, Optional, Protocol, Sequence, Tuple

import jax.numpy as jnp
import numpy as onp
from jax import tree_util
from totypes import json_utils, partition_utils, types

Array = jnp.ndarray | onp.ndarray[Any, Any]
PyTree = Any


@dataclasses.dataclass
class ParameterizedDensity2DArray:
    """Stores latents and metadata for a parameterized density array."""

    latents: "LatentsBase"
    metadata: Optional["MetadataBase"]


class LatentsBase:
    """Base class for latents of a parameterized density array."""

    pass


class MetadataBase:
    """Base class for metadata of a parameterized density array."""

    pass


tree_util.register_dataclass(
    ParameterizedDensity2DArray,
    data_fields=["latents", "metadata"],
    meta_fields=[],
)
json_utils.register_custom_type(ParameterizedDensity2DArray)


def partition_density_metadata(tree: PyTree) -> Tuple[PyTree, PyTree]:
    """Splits a pytree with parameterized densities into metadata from latents."""
    metadata, latents = partition_utils.partition(
        tree,
        select_fn=lambda x: isinstance(x, MetadataBase),
        is_leaf=_is_metadata_or_none,
    )
    return metadata, latents


def combine_density_metadata(metadata: PyTree, latents: PyTree) -> PyTree:
    """Combines pytrees containing metadata and latents."""
    return partition_utils.combine(metadata, latents, is_leaf=_is_metadata_or_none)


def _is_metadata_or_none(leaf: Any) -> bool:
    """Return `True` if `leaf` is `None` or density metadata."""
    return leaf is None or isinstance(leaf, MetadataBase)


@dataclasses.dataclass
class Density2DParameterization:
    """Stores `(from_density, to_density, constraints, update)` function triple."""

    from_density: "FromDensityFn"
    to_density: "ToDensityFn"
    constraints: "ConstraintsFn"
    update: "UpdateFn"


class FromDensityFn(Protocol):
    """Generate the latent representation of a density array."""

    def __call__(self, density: types.Density2DArray) -> ParameterizedDensity2DArray:
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

    def __call__(
        self,
        params: PyTree,
        updates: PyTree,
        value: jnp.ndarray,
        step: int,
    ) -> PyTree:
        ...


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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Density2DMetadata):
            return False
        if not (
            self.lower_bound == other.lower_bound
            and self.upper_bound == other.upper_bound
            and _arrays_equal_or_both_none(self.fixed_solid, other.fixed_solid)
            and _arrays_equal_or_both_none(self.fixed_void, other.fixed_void)
            and self.minimum_width == other.minimum_width
            and self.minimum_spacing == other.minimum_spacing
            and self.periodic == other.periodic
            and self.symmetries == other.symmetries
        ):
            return False
        return True

    @classmethod
    def from_density(self, density: types.Density2DArray) -> "Density2DMetadata":
        density_metadata_dict = dataclasses.asdict(density)
        del density_metadata_dict["array"]
        return Density2DMetadata(**density_metadata_dict)


def _arrays_equal_or_both_none(a: Optional[Array], b: Optional[Array]) -> bool:
    """Return `True` if `a` and `b` are equal arrays or both `None`."""
    if (a is None, b is None) not in ((True, True), (False, False)):
        return False
    if a is None and b is None:
        return True
    assert isinstance(a, onp.ndarray)
    assert isinstance(b, onp.ndarray)
    if a.dtype != b.dtype:
        return False
    if a.shape != b.shape:
        return False
    return bool(onp.all(a == b))


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
