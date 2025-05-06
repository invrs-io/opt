"""Defines a wrapper for optax optimizers.

Copyright (c) 2023 The INVRS-IO authors.
"""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import optax  # type: ignore[import-untyped]
from jax import tree_util
from totypes import types

from invrs_opt.optimizers import base
from invrs_opt.parameterization import (
    base as param_base,
    filter_project,
    gaussian_levelset,
    pixel,
)

PyTree = Any
WrappedOptaxState = Tuple[int, PyTree, PyTree, PyTree]


def wrapped_optax(opt: optax.GradientTransformation) -> base.Optimizer:
    """Return a wrapped optax optimizer."""
    return parameterized_wrapped_optax(
        opt=opt, penalty=0.0, density_parameterization=None
    )


def density_wrapped_optax(
    opt: optax.GradientTransformation,
    *,
    beta: float,
) -> base.Optimizer:
    """Wrapped optax optimizer with filter-project density parameterization.

    In the filter-project density parameterization, the optimization variable
    associated with a density array is a latent density array; the density is obtained
    by convolving (i.e. "filtering") the latent density with a Gaussian kernel having
    full-width at half-maximum equal to the length scale (the mean of declared minimum
    width and minimum spacing). Then, a tanh nonlinearity is used as a smooth threshold
    operation ("projection").

    Args:
        opt: The optax optimizer to be wrapped.
        beta: Determines the sharpness of the thresholding operation.

    Returns:
        The wrapped optax optimizer.
    """
    return parameterized_wrapped_optax(
        opt=opt,
        penalty=0.0,
        density_parameterization=filter_project.filter_project(beta=beta),
    )


def levelset_wrapped_optax(
    opt: optax.GradientTransformation,
    *,
    penalty: float,
    length_scale_spacing_factor: float = (
        gaussian_levelset.DEFAULT_LENGTH_SCALE_SPACING_FACTOR
    ),
    length_scale_fwhm_factor: float = (
        gaussian_levelset.DEFAULT_LENGTH_SCALE_FWHM_FACTOR
    ),
    length_scale_constraint_factor: float = (
        gaussian_levelset.DEFAULT_LENGTH_SCALE_CONSTRAINT_FACTOR
    ),
    smoothing_factor: int = gaussian_levelset.DEFAULT_SMOOTHING_FACTOR,
    length_scale_constraint_beta: float = (
        gaussian_levelset.DEFAULT_LENGTH_SCALE_CONSTRAINT_BETA
    ),
    length_scale_constraint_weight: float = (
        gaussian_levelset.DEFAULT_LENGTH_SCALE_CONSTRAINT_WEIGHT
    ),
    curvature_constraint_weight: float = (
        gaussian_levelset.DEFAULT_CURVATURE_CONSTRAINT_WEIGHT
    ),
    fixed_pixel_constraint_weight: float = (
        gaussian_levelset.DEFAULT_FIXED_PIXEL_CONSTRAINT_WEIGHT
    ),
    init_optimizer: optax.GradientTransformation = (
        gaussian_levelset.DEFAULT_INIT_OPTIMIZER
    ),
    init_steps: int = gaussian_levelset.DEFAULT_INIT_STEPS,
) -> base.Optimizer:
    """Wrapped optax optimizer with levelset density parameterization.

    In the levelset parameterization, the optimization variable associated with a
    density array is an array giving the amplitudes of Gaussian radial basis functions
    that represent a levelset function over the domain of the density. In the levelset
    parameterization, gradients are nonzero only at the edges of features, and in
    general the topology of a solution does not change during the course of
    optimization.

    The spacing and full-width at half-maximum of the Gaussian basis functions gives
    some amount of control over length scales. In addition, constraints associated with
    length scale, radius of curvature, and deviation from fixed pixels are
    automatically computed and penalized with a weight given by `penalty`. In general,
    this helps ensure that features in an optimized density array violate the specified
    constraints to a lesser degree. The constraints are based on "Analytical level set
    fabrication constraints for inverse design," by D. Vercruysse et al. (2019).

    Args:
        opt: The optax optimizer to be wrapped.
        penalty: The weight of the fabrication penalty, which combines length scale,
            curvature, and fixed pixel constraints.
        length_scale_spacing_factor: The number of levelset control points per unit of
            minimum length scale (mean of density minimum width and minimum spacing).
        length_scale_fwhm_factor: The ratio of Gaussian full-width at half-maximum to
            the minimum length scale.
        length_scale_constraint_factor: Multiplies the target length scale in the
            levelset constraints. A value greater than 1 is pessimistic and drives the
            solution to have a larger length scale (relative to smaller values).
        smoothing_factor: For values greater than 1, the density is initially computed
            at higher resolution and then downsampled, yielding smoother geometries.
        length_scale_constraint_beta: Controls relaxation of the length scale
            constraint near the zero level.
        length_scale_constraint_weight: The weight of the length scale constraint in
            the overall fabrication constraint peenalty.
        curvature_constraint_weight: The weight of the curvature constraint.
        fixed_pixel_constraint_weight: The weight of the fixed pixel constraint.
        init_optimizer: The optimizer used in the initialization of the levelset
            parameterization. At initialization, the latent parameters are optimized so
            that the initial parameters match the binarized initial density.
        init_steps: The number of optimization steps used in the initialization.

    Returns:
        The wrapped optax optimizer.
    """
    return parameterized_wrapped_optax(
        opt=opt,
        penalty=penalty,
        density_parameterization=gaussian_levelset.gaussian_levelset(
            length_scale_spacing_factor=length_scale_spacing_factor,
            length_scale_fwhm_factor=length_scale_fwhm_factor,
            length_scale_constraint_factor=length_scale_constraint_factor,
            smoothing_factor=smoothing_factor,
            length_scale_constraint_beta=length_scale_constraint_beta,
            length_scale_constraint_weight=length_scale_constraint_weight,
            curvature_constraint_weight=curvature_constraint_weight,
            fixed_pixel_constraint_weight=fixed_pixel_constraint_weight,
            init_optimizer=init_optimizer,
            init_steps=init_steps,
        ),
    )


# -----------------------------------------------------------------------------
# Base parameterized wrapped optax optimizer.
# -----------------------------------------------------------------------------


def parameterized_wrapped_optax(
    opt: optax.GradientTransformation,
    density_parameterization: Optional[param_base.Density2DParameterization],
    penalty: float,
) -> base.Optimizer:
    """Wrapped optax optimizer with specified density parameterization.

    Args:
        opt: The optax `GradientTransformation` to be wrapped.
        density_parameterization: The parameterization to be used, or `None`. When no
            parameterization is given, the direct pixel parameterization is used for
            density arrays.
        penalty: The weight of the scalar penalty formed from the constraints of the
            parameterization.

    Returns:
        The `base.Optimizer`.
    """

    if density_parameterization is None:
        density_parameterization = pixel.pixel()

    def init_fn(params: PyTree) -> WrappedOptaxState:
        """Initializes the optimization state."""
        latent_params = _init_latents(params)
        _, latents = param_base.partition_density_metadata(latent_params)
        return (
            0,  # step
            _params_from_latent_params(latent_params),  # params
            latent_params,  # latent params
            opt.init(latents),  # opt state
        )

    def params_fn(state: WrappedOptaxState) -> PyTree:
        """Returns the parameters for the given `state`."""
        _, params, _, _ = state
        return params

    def update_fn(
        *,
        grad: PyTree,
        value: jnp.ndarray,
        params: PyTree,
        state: WrappedOptaxState,
    ) -> WrappedOptaxState:
        """Updates the state."""
        del params

        step, params, latent_params, opt_state = state
        metadata, latents = param_base.partition_density_metadata(latent_params)

        def _params_from_latents(latents: PyTree) -> PyTree:
            latent_params = param_base.combine_density_metadata(metadata, latents)
            return _params_from_latent_params(latent_params)

        def _constraint_loss_latents(latents: PyTree) -> jnp.ndarray:
            latent_params = param_base.combine_density_metadata(metadata, latents)
            return _constraint_loss(latent_params)

        _, vjp_fn = jax.vjp(_params_from_latents, latents)
        (latents_grad,) = vjp_fn(grad)

        if not (
            tree_util.tree_structure(latents_grad) == tree_util.tree_structure(latents)
        ):
            raise ValueError(
                f"Tree structure of `latents_grad` was different than expected, got \n"
                f"{tree_util.tree_structure(latents_grad)} but expected \n"
                f"{tree_util.tree_structure(latents)}."
            )

        constraint_loss_grad = jax.grad(_constraint_loss_latents)(latents)
        latents_grad = tree_util.tree_map(
            lambda a, b: a + b, latents_grad, constraint_loss_grad
        )

        latent_updates, opt_state = opt.update(latents_grad, opt_state, params=latents)
        latent_params = _apply_updates(
            params=latent_params,
            updates=param_base.combine_density_metadata(metadata, latent_updates),
            value=value,
            step=step,
        )
        latent_params = _clip(latent_params)
        params = _params_from_latent_params(latent_params)
        return (step + 1, params, latent_params, opt_state)

    # -------------------------------------------------------------------------
    # Functions related to the density parameterization.
    # -------------------------------------------------------------------------

    def _init_latents(params: PyTree) -> PyTree:
        def _leaf_init_latents(leaf: Any) -> Any:
            leaf = _clip(leaf)
            if not _is_density(leaf):
                return leaf
            return density_parameterization.from_density(leaf)

        return tree_util.tree_map(_leaf_init_latents, params, is_leaf=_is_custom_type)

    def _params_from_latent_params(params: PyTree) -> PyTree:
        def _leaf_params_from_latents(leaf: Any) -> Any:
            if not _is_parameterized_density(leaf):
                return leaf
            return density_parameterization.to_density(leaf)

        return tree_util.tree_map(
            _leaf_params_from_latents,
            params,
            is_leaf=_is_parameterized_density,
        )

    def _apply_updates(
        params: PyTree,
        updates: PyTree,
        value: jnp.ndarray,
        step: int,
    ) -> PyTree:
        def _leaf_apply_updates(update: Any, leaf: Any) -> Any:
            if _is_parameterized_density(leaf):
                return density_parameterization.update(
                    params=leaf, updates=update, value=value, step=step
                )
            else:
                return optax.apply_updates(params=leaf, updates=update)

        return tree_util.tree_map(
            _leaf_apply_updates,
            updates,
            params,
            is_leaf=_is_parameterized_density,
        )

    # -------------------------------------------------------------------------
    # Functions related to the constraints to be minimized.
    # -------------------------------------------------------------------------

    def _constraint_loss(latent_params: PyTree) -> jnp.ndarray:
        def _constraint_loss_leaf(
            leaf: param_base.ParameterizedDensity2DArray,
        ) -> jnp.ndarray:
            constraints = density_parameterization.constraints(leaf)
            constraints = tree_util.tree_map(
                lambda x: jnp.sum(jnp.maximum(x, 0.0) ** 2),
                constraints,
            )
            return jnp.sum(jnp.asarray(constraints))

        losses = [0.0] + [
            _constraint_loss_leaf(p)
            for p in tree_util.tree_leaves(
                latent_params, is_leaf=_is_parameterized_density
            )
            if _is_parameterized_density(p)
        ]
        return penalty * jnp.sum(jnp.asarray(losses))

    return base.Optimizer(init=init_fn, params=params_fn, update=update_fn)


def _is_density(leaf: Any) -> Any:
    """Return `True` if `leaf` is a density array."""
    return isinstance(leaf, types.Density2DArray)


def _is_parameterized_density(leaf: Any) -> Any:
    """Return `True` if `leaf` is a parameterized density array."""
    return isinstance(leaf, param_base.ParameterizedDensity2DArray)


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
