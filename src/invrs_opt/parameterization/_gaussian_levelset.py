"""Defines Gaussian radial basis function levelset parameterization.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import optax  # type: ignore[import-untyped]
from jax import tree_util
from totypes import symmetry, types

from invrs_opt.parameterization import base, transforms

PyTree = Any


@dataclasses.dataclass
class GaussianLevelsetParams(base.ParameterizedDensity2DArrayBase):
    """Parameters of a density represented by a Gaussian levelset.

    Attributes:
        amplitude:
        length_scale_spacing_factor:
        length_scale_fwhm_factor:
        length_scale_constraint_factor:
        smoothing_factor:
        density_shape:
        density_metadata:
    """

    amplitude: jnp.ndarray
    length_scale_spacing_factor: float
    length_scale_fwhm_factor: float
    length_scale_constraint_factor: float
    smoothing_factor: int
    density_shape: Tuple[int, ...]
    density_metadata: base.Density2DMetadata

    def example_density(self) -> types.Density2DArray:
        with jax.ensure_compile_time_eval():
            return types.Density2DArray(
                array=jnp.zeros(self.density_shape),
                **dataclasses.asdict(self.density_metadata),
            )


tree_util.register_dataclass(
    GaussianLevelsetParams,
    data_fields=["amplitude"],
    meta_fields=[
        "length_scale_spacing_factor",
        "length_scale_fwhm_factor",
        "length_scale_constraint_factor",
        "smoothing_factor",
        "density_shape",
        "density_metadata",
    ],
)


def gaussian_levelset(
    *,
    length_scale_spacing_factor: float = 2.0,
    length_scale_fwhm_factor: float = 1.0,
    length_scale_constraint_factor: float = 1.3,
    smoothing_factor: int = 2,
    constraint_beta: float = 0.333,
    init_steps: int = 50,
    init_optimizer: optax.GradientTransformation = optax.adam(1e-1),
) -> base.Density2DParameterization:
    """Defines a levelset parameterization with Gaussian radial basis functions.

    Args:
        length_scale_spacing_factor:
        length_scale_fwhm_factor:
        length_scale_constraint_factor:
        smoothing_factor:
        constraint_beta:
        init_steps:
        init_optimizer:

    Returns:
        The `Density2DParameterization`.
    """

    def from_density_fn(density: types.Density2DArray) -> GaussianLevelsetParams:
        """Return level set parameters for the given `density`."""
        length_scale = (density.minimum_width + density.minimum_spacing) / 2
        shape = density.shape[:-2] + (
            int(density.shape[-2] / length_scale * length_scale_spacing_factor),
            int(density.shape[-1] / length_scale * length_scale_spacing_factor),
        )

        target_array = transforms.apply_fixed_pixels(density).array
        target_array = jnp.sign(target_array)

        # Generate the initial amplitude array.
        amplitude = density.array - (density.upper_bound + density.lower_bound) / 2
        amplitude = jnp.sign(amplitude)
        amplitude = transforms.resample(amplitude, shape)

        # If the density is not periodic, ensure there are level set control points
        # beyond the edge of the density array.
        pad_width = ((0, 0),) * (amplitude.ndim - 2)
        pad_width += ((0, 0),) if density.periodic[0] else ((1, 1),)
        pad_width += ((0, 0),) if density.periodic[1] else ((1, 1),)
        amplitude = jnp.pad(amplitude, pad_width, mode="edge")

        density_metadata_dict = dataclasses.asdict(density)
        del density_metadata_dict["array"]
        density_metadata = base.Density2DMetadata(**density_metadata_dict)
        params = GaussianLevelsetParams(
            amplitude=amplitude,
            length_scale_spacing_factor=length_scale_spacing_factor,
            length_scale_fwhm_factor=length_scale_fwhm_factor,
            length_scale_constraint_factor=length_scale_constraint_factor,
            smoothing_factor=smoothing_factor,
            density_shape=density.shape,
            density_metadata=density_metadata,
        )

        def step_fn(
            _: int,
            params_and_state: Tuple[PyTree, PyTree],
        ) -> Tuple[PyTree, PyTree]:
            def loss_fn(params: GaussianLevelsetParams) -> jnp.ndarray:
                density_from_params = to_density_fn(params, mask_gradient=False)
                return jnp.mean((density_from_params.array - target_array) ** 2)

            params, state = params_and_state
            grad = jax.grad(loss_fn)(params)
            updates, state = init_optimizer.update(grad, params=params, state=state)
            params = optax.apply_updates(params, updates)
            return params, state

        state = init_optimizer.init(params)
        params, state = jax.lax.fori_loop(
            0, init_steps, body_fun=step_fn, init_val=(params, state)
        )
        maxval = jnp.amax(jnp.abs(params.amplitude), axis=(-2, -1), keepdims=True)
        return dataclasses.replace(params, amplitude=params.amplitude / maxval)

    def to_density_fn(
        params: GaussianLevelsetParams,
        mask_gradient: bool = True,
    ) -> types.Density2DArray:
        """Return a density from the latent parameters."""
        example_density = params.example_density()
        periodic: Tuple[bool, bool] = example_density.periodic
        phi = _phi_from_params(params=params, pad_pixels=0)
        array = _levelset_threshold(
            phi=phi,
            periodic=periodic,
            mask_gradient=mask_gradient,
        )
        array = transforms.downsample_spatial_dims(array, smoothing_factor)

        lb = example_density.lower_bound
        ub = example_density.upper_bound
        array = lb + array * (ub - lb)
        assert array.shape == example_density.shape
        return dataclasses.replace(example_density, array=array)

    def constraints_fn(params: GaussianLevelsetParams) -> jnp.ndarray:
        """Computes constraints associated with the params."""
        example_density = params.example_density()
        minimum_length_scale = 0.5 * (
            example_density.minimum_width + example_density.minimum_spacing
        )
        return jnp.stack(
            _levelset_constraints(
                params,
                minimum_length_scale=minimum_length_scale,
                beta=constraint_beta,
            ),
            axis=-1,
        )

    return base.Density2DParameterization(
        to_density=to_density_fn,
        from_density=from_density_fn,
        constraints=constraints_fn,
    )


# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------


def _phi_from_params(
    params: GaussianLevelsetParams,
    pad_pixels: int = 0,
) -> jnp.ndarray:
    """Return the level set function for the given `params`."""
    example_density = params.example_density()
    # Coordinates on which the level set function is to be evaluated. These may
    # differ from the pixel coordinates of the density, if `smoothing_factor > 1`.
    num_i = example_density.shape[-2] * params.smoothing_factor
    num_j = example_density.shape[-1] * params.smoothing_factor
    highres_i, highres_j = jnp.meshgrid(
        (0.5 + jnp.arange(-pad_pixels, num_i + pad_pixels)) / params.smoothing_factor,
        (0.5 + jnp.arange(-pad_pixels, num_j + pad_pixels)) / params.smoothing_factor,
        indexing="ij",
    )
    highres_i = highres_i[..., jnp.newaxis, jnp.newaxis]
    highres_j = highres_j[..., jnp.newaxis, jnp.newaxis]

    # Coordinates for the control points of the Gaussian radial basis functions.
    amplitude = params.amplitude
    levelset_i, levelset_j = _control_point_coords(
        density_shape=params.density_shape[-2:],  # type: ignore[arg-type]
        levelset_shape=amplitude.shape[-2:],  # type: ignore[arg-type]
        periodic=example_density.periodic,
    )

    # Handle periodicity by replicating control points over a 3x3 supercell.
    if example_density.periodic[0]:
        levelset_i = jnp.concatenate(
            [
                levelset_i - example_density.shape[-2],
                levelset_i,
                levelset_i + example_density.shape[-2],
            ],
            axis=-2,
        )
        levelset_j = jnp.concatenate([levelset_j] * 3, axis=-2)
        amplitude = jnp.concat([amplitude] * 3, axis=-2)
    if example_density.periodic[1]:
        levelset_i = jnp.concatenate([levelset_i] * 3, axis=-1)
        levelset_j = jnp.concatenate(
            [
                levelset_j - example_density.shape[-1],
                levelset_j,
                levelset_j + example_density.shape[-1],
            ],
            axis=-1,
        )
        amplitude = jnp.concat([amplitude] * 3, axis=-1)

    distance = jnp.sqrt((highres_i - levelset_i) ** 2 + (highres_j - levelset_j) ** 2)
    length_scale = 0.5 * (
        example_density.minimum_width + example_density.minimum_spacing
    )
    fwhm = length_scale * params.length_scale_fwhm_factor
    sigma = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
    array = jnp.exp(-(distance**2) / (2 * sigma**2))
    array *= amplitude[..., jnp.newaxis, jnp.newaxis, :, :]
    array = jnp.sum(array, axis=(-2, -1))
    assert array.shape[-2:] == highres_i.shape[:-2]
    assert array.shape[-2] % params.smoothing_factor == 0
    assert array.shape[-1] % params.smoothing_factor == 0
    array = symmetry.symmetrize(array, tuple(example_density.symmetries))
    return array


def _phi_derivatives_and_radius(
    params: GaussianLevelsetParams,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,]:
    """Compute the levelset function and its first and second derivatives."""

    # Compute phi with padding to ensure accurate gradient at edge of array.
    pad_pixels = 6
    phi = _phi_from_params(params=params, pad_pixels=pad_pixels)

    d = 1 / params.smoothing_factor
    phi_x, phi_y = jnp.gradient(phi, d, axis=(-2, -1))
    phi_xx, phi_yx = jnp.gradient(phi_x, d, axis=(-2, -1))
    phi_xy, phi_yy = jnp.gradient(phi_y, d, axis=(-2, -1))

    phi_v = _sqrt_safe(phi_x**2 + phi_y**2)

    phi_v_near_zero = jnp.isclose(phi_v, 0.0)
    phi_v_squared_safe = jnp.where(phi_v_near_zero, 1.0, phi_v**2)
    weight_xx = phi_x**2 / phi_v_squared_safe
    weight_yy = phi_y**2 / phi_v_squared_safe
    weight_xy = (phi_x * phi_y) / phi_v_squared_safe
    phi_vv = weight_xx * phi_xx + weight_xy * (phi_xy + phi_yx) + weight_yy * phi_yy

    qx: jnp.ndarray = jnp.gradient(  # type: ignore[assignment]
        phi_x / jnp.abs(phi_v), d, axis=-2
    )
    qy: jnp.ndarray = jnp.gradient(  # type: ignore[assignment]
        phi_y / jnp.abs(phi_v), d, axis=-1
    )
    radius = 1 / (qx + qy)

    def _unpad(x: jnp.ndarray) -> jnp.ndarray:
        return x[..., pad_pixels:-pad_pixels, pad_pixels:-pad_pixels]

    return _unpad(phi), _unpad(phi_v), _unpad(phi_vv), _unpad(radius)


def _levelset_constraints(
    params: GaussianLevelsetParams,
    minimum_length_scale: float,
    beta: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute constraints for minimum width and minimum spacing.

    The constraints are based on "Analytical level set fabrication constraints for
    inverse design," by D. Vercruysse et al. (2019). Constraints are satisfied when
    they are non-positive.

    https://www.nature.com/articles/s41598-019-45026-0

    Args:
        params: The parameters of the Gaussian levelset.
        beta: Parameter which relaxes the constraint near the zero-plane.

    Returns:
        The minimum length scale and minimum curvature constraint arrays.
    """
    phi, phi_v, phi_vv, radius = _phi_derivatives_and_radius(params)

    d = minimum_length_scale * params.length_scale_constraint_factor
    length_scale_constraint = (
        jnp.abs(phi_vv) / (jnp.pi / d * jnp.abs(phi) + beta * phi_v) - jnp.pi / d
    )
    curvature_constraint = jnp.abs(1 / radius * jnp.arctan(phi_v / phi)) - jnp.pi / d

    return length_scale_constraint, curvature_constraint


# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------


def _control_point_coords(
    density_shape: Tuple[int, int],
    levelset_shape: Tuple[int, int],
    periodic: Tuple[
        bool,
        bool,
    ],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the control point coordinates."""
    # If the levelset is periodic along any axis, the first and last control
    # points along that axis lie outside the bounds of the density.
    offset_i = 0.5 if periodic[0] else -0.5
    offset_j = 0.5 if periodic[1] else -0.5
    range_i = levelset_shape[-2] - (0 if periodic[0] else 2)
    range_j = levelset_shape[-1] - (0 if periodic[1] else 2)

    factor_i = density_shape[-2] / range_i
    factor_j = density_shape[-1] / range_j
    levelset_i, levelset_j = jnp.meshgrid(
        (offset_i + jnp.arange(levelset_shape[-2])) * factor_i,
        (offset_j + jnp.arange(levelset_shape[-1])) * factor_j,
        indexing="ij",
    )
    return levelset_i, levelset_j


def _sqrt_safe(x: jnp.ndarray) -> jnp.ndarray:
    x_near_zero = jnp.isclose(x, 0.0)
    x_safe = jnp.where(x_near_zero, 1, x)
    return jnp.where(x_near_zero, 0.0, jnp.sqrt(x_safe))


def _levelset_threshold(
    phi: jnp.ndarray,
    periodic: Tuple[bool, bool],
    mask_gradient: bool,
) -> jnp.ndarray:
    """Thresholds a level set function `phi`."""
    if mask_gradient:
        interface = transforms.interface_pixels(phi, periodic)
        phi = jnp.where(interface, phi, jax.lax.stop_gradient(phi))
    thresholded = (phi > 0).astype(float) + (phi - jax.lax.stop_gradient(phi))
    return thresholded
