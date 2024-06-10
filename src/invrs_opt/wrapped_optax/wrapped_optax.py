import dataclasses
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import tree_util
from totypes import types

from invrs_opt import base, transform

PyTree = Any
WrappedOptaxState = Tuple[PyTree, PyTree, PyTree]


def wrapped_optax(opt: optax.GradientTransformation) -> base.Optimizer:
    """Return a wrapped optax optimizer."""
    return transformed_wrapped_optax(
        opt=opt,
        transform_fn=lambda x: x,
        initialize_latent_fn=lambda x: x,
    )


def density_wrapped_optax(
    opt: optax.GradientTransformation,
    beta: float,
) -> base.Optimizer:
    """Return a wrapped optax optimizer with transforms for density arrays."""

    def transform_fn(tree: PyTree) -> PyTree:
        return tree_util.tree_map(
            lambda x: transform_density(x) if _is_density(x) else x,
            tree,
            is_leaf=_is_density,
        )

    def initialize_latent_fn(tree: PyTree) -> PyTree:
        return tree_util.tree_map(
            lambda x: initialize_latent_density(x) if _is_density(x) else x,
            tree,
            is_leaf=_is_density,
        )

    def transform_density(density: types.Density2DArray) -> types.Density2DArray:
        transformed = types.symmetrize_density(density)
        transformed = transform.density_gaussian_filter_and_tanh(transformed, beta=beta)
        # Scale to ensure that the full valid range of the density array is reachable.
        mid_value = (density.lower_bound + density.upper_bound) / 2
        transformed = tree_util.tree_map(
            lambda array: mid_value + (array - mid_value) / jnp.tanh(beta), transformed
        )
        return transform.apply_fixed_pixels(transformed)

    def initialize_latent_density(
        density: types.Density2DArray,
    ) -> types.Density2DArray:
        array = transform.normalized_array_from_density(density)
        array = jnp.clip(array, -1, 1)
        array *= jnp.tanh(beta)
        latent_array = jnp.arctanh(array) / beta
        latent_array = transform.rescale_array_for_density(latent_array, density)
        return dataclasses.replace(density, array=latent_array)

    return transformed_wrapped_optax(
        opt=opt,
        transform_fn=transform_fn,
        initialize_latent_fn=initialize_latent_fn,
    )


def transformed_wrapped_optax(
    opt: optax.GradientTransformation,
    transform_fn: Callable[[PyTree], PyTree],
    initialize_latent_fn: Callable[[PyTree], PyTree],
) -> base.Optimizer:
    """Return a wrapped optax optimizer for transformed latent parameters.

    Args:
        opt: The optax `GradientTransformation` to be wrapped.
        transform_fn: Function which transforms the internal latent parameters to
            the parameters returned by the optimizer.
        initialize_latent_fn: Function which computes the initial latent parameters
            given the initial parameters.

    Returns:
        The `base.Optimizer`.
    """

    def init_fn(params: PyTree) -> WrappedOptaxState:
        """Initializes the optimization state."""
        latent_params = initialize_latent_fn(_clip(params))
        params = transform_fn(latent_params)
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
        del value

        _, latent_params, opt_state = state
        _, vjp_fn = jax.vjp(transform_fn, latent_params)
        (latent_grad,) = vjp_fn(grad)

        updates, opt_state = opt.update(latent_grad, opt_state)
        latent_params = optax.apply_updates(params=latent_params, updates=updates)
        latent_params = _clip(latent_params)
        params = transform_fn(latent_params)
        return params, latent_params, opt_state

    return base.Optimizer(
        init=init_fn,
        params=params_fn,
        update=update_fn,
    )


def _is_density(leaf: Any) -> Any:
    """Return `True` if `leaf` is a density array."""
    return isinstance(leaf, types.Density2DArray)


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
