"""Defines a wrapper for optax optimizers.

Copyright (c) 2023 The INVRS-IO authors.
"""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import optax  # type: ignore[import-untyped]
from jax import tree_util
from totypes import types

from invrs_opt import parameterization
from invrs_opt.optimizers import base
from invrs_opt.parameterization import base as parameterization_base

PyTree = Any
WrappedOptaxState = Tuple[PyTree, PyTree, PyTree]


def wrapped_optax(opt: optax.GradientTransformation) -> base.Optimizer:
    """Return a wrapped optax optimizer."""
    return parameterized_wrapped_optax(opt=opt, density_parameterization=None)


def parameterized_wrapped_optax(
    opt: optax.GradientTransformation,
    density_parameterization: Optional[parameterization_base.Density2DParameterization],
) -> base.Optimizer:
    """Return a wrapped optax optimizer for transformed latent parameters.

    Args:
        opt: The optax `GradientTransformation` to be wrapped.
        density_parameterization: The parameterization used for `Density2DArray` types.

    Returns:
        The `base.Optimizer`.
    """

    if density_parameterization is None:
        density_parameterization = parameterization.pixel()

    def _init_latents(params: PyTree) -> PyTree:
        def _leaf_init_latents(leaf: Any) -> Any:
            leaf = _clip(leaf)
            if not _is_density(leaf):
                return leaf
            return density_parameterization.from_density(leaf)

        return tree_util.tree_map(_leaf_init_latents, params, is_leaf=_is_custom_type)

    def _params_from_latents(params: PyTree) -> PyTree:
        def _leaf_params_from_latents(leaf: Any) -> Any:
            if not _is_parameterized_density(leaf):
                return leaf
            return density_parameterization.to_density(leaf)

        return tree_util.tree_map(
            _leaf_params_from_latents,
            params,
            is_leaf=_is_parameterized_density,
        )

    def init_fn(params: PyTree) -> WrappedOptaxState:
        """Initializes the optimization state."""
        latent_params = _init_latents(params)
        params = _params_from_latents(latent_params)
        return params, latent_params, opt.init(latent_params)

    def params_fn(state: WrappedOptaxState) -> PyTree:
        """Returns the parameters for the given `state`."""
        params, _, _ = state
        return params

    def update_fn(
        *,
        grad: PyTree,
        value: float,
        params: PyTree,
        state: WrappedOptaxState,
    ) -> WrappedOptaxState:
        """Updates the state."""
        del value, params

        _, latent_params, opt_state = state
        _, vjp_fn = jax.vjp(_params_from_latents, latent_params)
        (latent_grad,) = vjp_fn(grad)

        updates, opt_state = opt.update(
            updates=latent_grad, state=opt_state, params=latent_params
        )
        latent_params = optax.apply_updates(params=latent_params, updates=updates)
        latent_params = _clip(latent_params)
        params = _params_from_latents(latent_params)
        return params, latent_params, opt_state

    return base.Optimizer(init=init_fn, params=params_fn, update=update_fn)


def _is_density(leaf: Any) -> Any:
    """Return `True` if `leaf` is a density array."""
    return isinstance(leaf, types.Density2DArray)


def _is_parameterized_density(leaf: Any) -> Any:
    """Return `True` if `leaf` is a parameterized density array."""
    return isinstance(leaf, parameterization_base.ParameterizedDensity2DArrayBase)


def _is_custom_type(leaf: Any) -> bool:
    """Return `True` if `leaf` is a recognized custom type."""
    return isinstance(leaf, (types.BoundedArray, types.Density2DArray))


def _clip(pytree: PyTree) -> PyTree:
    """Clips leaves on `pytree` to their bounds."""

    def _clip_fn(leaf: Any) -> Any:
        if not _is_custom_type(leaf):
            return leaf
        if leaf.lower_bound is None and leaf.upper_bound is None:
            return leaf
        return tree_util.tree_map(
            lambda x: jnp.clip(x, leaf.lower_bound, leaf.upper_bound), leaf
        )

    return tree_util.tree_map(_clip_fn, pytree, is_leaf=_is_custom_type)
