"""Defines Gaussian radial basis function levelset parameterization.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import optax  # type: ignore[import-untyped]
from jax import tree_util
from totypes import json_utils, symmetry, types

from invrs_opt.parameterization import base, transforms

PyTree = Any

DEFAULT_LENGTH_SCALE_SPACING_FACTOR: float = 2.0
DEFAULT_LENGTH_SCALE_FWHM_FACTOR: float = 1.0
DEFAULT_LENGTH_SCALE_CONSTRAINT_FACTOR: float = 1.15
DEFAULT_SMOOTHING_FACTOR: int = 2
DEFAULT_LENGTH_SCALE_CONSTRAINT_BETA: float = 0.333
DEFAULT_LENGTH_SCALE_CONSTRAINT_WEIGHT: float = 1.0
DEFAULT_CURVATURE_CONSTRAINT_WEIGHT: float = 2.0
DEFAULT_FIXED_PIXEL_CONSTRAINT_WEIGHT: float = 10.0
DEFAULT_INIT_STEPS: int = 50
DEFAULT_INIT_OPTIMIZER: optax.GradientTransformation = optax.adam(1e-1)


@dataclasses.dataclass
class GaussianLevelsetParams(base.ParameterizedDensity2DArray):
    """Stores parameters for the Gaussian levelset parameterization."""

    latents: "GaussianLevelsetLatents"
    metadata: "GaussianLevelsetMetadata"


@dataclasses.dataclass
class GaussianLevelsetLatents(base.LatentsBase):
    """Stores latent parameters for the Gaussian levelset parameterization.

    Attributes:
        amplitude: Array giving the amplitude of the Gaussian basis function at
            levelset control points.
    """

    amplitude: jnp.ndarray


@dataclasses.dataclass
class GaussianLevelsetMetadata(base.MetadataBase):
    """Stores metadata for the Gaussian levelset parameterization.

    Attributes:
        length_scale_spacing_factor: The number of levelset control points per unit of
            minimum length scale (mean of density minimum width and minimum spacing).
        length_scale_fwhm_factor: The ratio of Gaussian full-width at half-maximum to
            the minimum length scale.
        smoothing_factor: For values greater than 1, the density is initially computed
            at higher resolution and then downsampled, yielding smoother geometries.
        density_shape: Shape of the density array obtained from the parameters.
        density_metadata: Metadata for the density array obtained from the parameters.
    """

    length_scale_spacing_factor: float
    length_scale_fwhm_factor: float
    smoothing_factor: int
    density_shape: Tuple[int, ...]
    density_metadata: base.Density2DMetadata

    def __post_init__(self) -> None:
        self.density_shape = tuple(self.density_shape)


tree_util.register_dataclass(
    GaussianLevelsetParams,
    data_fields=["latents", "metadata"],
    meta_fields=[],
)
tree_util.register_dataclass(
    GaussianLevelsetLatents,
    data_fields=["amplitude"],
    meta_fields=[],
)
tree_util.register_dataclass(
    GaussianLevelsetMetadata,
    data_fields=[
        "length_scale_spacing_factor",
        "length_scale_fwhm_factor",
        "density_metadata",
    ],
    meta_fields=["density_shape", "smoothing_factor"],
)
json_utils.register_custom_type(GaussianLevelsetParams)
json_utils.register_custom_type(GaussianLevelsetLatents)
json_utils.register_custom_type(GaussianLevelsetMetadata)


def gaussian_levelset(
    *,
    length_scale_spacing_factor: float = DEFAULT_LENGTH_SCALE_SPACING_FACTOR,
    length_scale_fwhm_factor: float = DEFAULT_LENGTH_SCALE_FWHM_FACTOR,
    length_scale_constraint_factor: float = DEFAULT_LENGTH_SCALE_CONSTRAINT_FACTOR,
    smoothing_factor: int = DEFAULT_SMOOTHING_FACTOR,
    length_scale_constraint_beta: float = DEFAULT_LENGTH_SCALE_CONSTRAINT_BETA,
    length_scale_constraint_weight: float = DEFAULT_LENGTH_SCALE_CONSTRAINT_WEIGHT,
    curvature_constraint_weight: float = DEFAULT_CURVATURE_CONSTRAINT_WEIGHT,
    fixed_pixel_constraint_weight: float = DEFAULT_FIXED_PIXEL_CONSTRAINT_WEIGHT,
    init_optimizer: optax.GradientTransformation = DEFAULT_INIT_OPTIMIZER,
    init_steps: int = DEFAULT_INIT_STEPS,
) -> base.Density2DParameterization:
    """Defines a levelset parameterization with Gaussian radial basis functions.

    Args:
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
        The `Density2DParameterization`.
    """

    def from_density_fn(density: types.Density2DArray) -> GaussianLevelsetParams:
        """Return level set parameters for the given `density`."""
        density.array = jnp.clip(
            density.array, min=density.lower_bound, max=density.upper_bound
        )
        length_scale = (density.minimum_width + density.minimum_spacing) / 2
        spacing_factor = length_scale_spacing_factor / length_scale
        shape = density.shape[:-2] + (
            int(jnp.ceil(density.shape[-2] * spacing_factor)),
            int(jnp.ceil(density.shape[-1] * spacing_factor)),
        )

        mid_value = 0.5 * (density.lower_bound + density.upper_bound)
        value_range = density.upper_bound - density.lower_bound
        target_array = transforms.apply_fixed_pixels(density).array
        target_array = (
            jnp.sign(target_array - mid_value) * 0.5 * value_range + mid_value
        )

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

        latents = GaussianLevelsetLatents(amplitude=amplitude)
        metadata = GaussianLevelsetMetadata(
            length_scale_spacing_factor=length_scale_spacing_factor,
            length_scale_fwhm_factor=length_scale_fwhm_factor,
            smoothing_factor=smoothing_factor,
            density_shape=density.shape,
            density_metadata=base.Density2DMetadata.from_density(density),
        )

        def step_fn(
            _: int,
            params_and_state: Tuple[PyTree, PyTree],
        ) -> Tuple[PyTree, PyTree]:
            def loss_fn(latents: GaussianLevelsetLatents) -> jnp.ndarray:
                params = GaussianLevelsetParams(latents, metadata=metadata)
                density_from_params = to_density_fn(params, mask_gradient=False)
                return jnp.mean((density_from_params.array - target_array) ** 2)

            params, state = params_and_state
            grad = jax.grad(loss_fn)(params)
            updates, state = init_optimizer.update(grad, params=params, state=state)
            params = optax.apply_updates(params, updates)
            return params, state

        state = init_optimizer.init(latents)
        latents, _ = jax.lax.fori_loop(
            0, init_steps, body_fun=step_fn, init_val=(latents, state)
        )

        maxval = jnp.amax(jnp.abs(latents.amplitude), axis=(-2, -1), keepdims=True)
        latents = dataclasses.replace(latents, amplitude=latents.amplitude / maxval)
        return GaussianLevelsetParams(latents=latents, metadata=metadata)

    def to_density_fn(
        params: GaussianLevelsetParams,
        mask_gradient: bool = True,
    ) -> types.Density2DArray:
        """Return a density from the latent parameters."""
        array = _to_array(params, mask_gradient=mask_gradient, pad_pixels=0)

        example_density = _example_density(params)
        lb = example_density.lower_bound
        ub = example_density.upper_bound
        array = lb + array * (ub - lb)
        assert array.shape == example_density.shape
        return dataclasses.replace(example_density, array=array)

    def constraints_fn(
        params: GaussianLevelsetParams,
        mask_gradient: bool = True,
        pad_pixels: int = 2,
    ) -> jnp.ndarray:
        """Computes constraints associated with the params."""
        return analytical_constraints(
            params=params,
            length_scale_constraint_factor=length_scale_constraint_factor,
            length_scale_constraint_beta=length_scale_constraint_beta,
            length_scale_constraint_weight=length_scale_constraint_weight,
            curvature_constraint_weight=curvature_constraint_weight,
            fixed_pixel_constraint_weight=fixed_pixel_constraint_weight,
            mask_gradient=mask_gradient,
            pad_pixels=pad_pixels,
        )

    def update_fn(
        params: GaussianLevelsetParams,
        updates: GaussianLevelsetParams,
        value: jnp.ndarray,
        step: int,
    ) -> GaussianLevelsetParams:
        """Perform updates to `params` required for the given `step`."""
        del step, value
        return GaussianLevelsetParams(
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


# -----------------------------------------------------------------------------
# Functions to obtain arrays from the levelset parameterization.
# -----------------------------------------------------------------------------


def _example_density(params: GaussianLevelsetParams) -> types.Density2DArray:
    """Returns an example density with appropriate shape and metadata."""
    with jax.ensure_compile_time_eval():
        return types.Density2DArray(
            array=jnp.zeros(params.metadata.density_shape),
            **dataclasses.asdict(params.metadata.density_metadata),
        )


def _to_array(
    params: GaussianLevelsetParams,
    mask_gradient: bool,
    pad_pixels: int,
) -> jnp.ndarray:
    """Return an array from the parameters.

    The array has a value of `1` where the levelset array is positive, and a value
    of `-1` elsewhere. The final density array can be obtained by rescaling this array
    to have the appropriate upper and lower bounds.

    Args:
        params: The parameters from which the density is obtained.
        mask_gradient: If `True`, the gradient is masked so that it is nonzero only at
            the borders of features.
        pad_pixels: A non-negative integer giving the additional pixels to be included
            beyond the boundaries of the parameterized density.

    Returns:
        The array.
    """
    example_density = _example_density(params)
    periodic: Tuple[bool, bool] = example_density.periodic
    phi = _phi_from_params(params=params, pad_pixels=pad_pixels)
    array = _levelset_threshold(phi=phi, periodic=periodic, mask_gradient=mask_gradient)
    return _downsample_spatial_dims(array, params.metadata.smoothing_factor)


def _phi_from_params(
    params: GaussianLevelsetParams,
    pad_pixels: int,
) -> jnp.ndarray:
    """Return the levelset function for the given `params`.

    Args:
        params: The parameters from which the density is obtained.
        pad_pixels: A non-negative integer giving the additional pixels to be included
            beyond the boundaries of the parameterized density.

    Returns:
        The levelset array `phi`.
    """
    with jax.ensure_compile_time_eval():
        example_density = _example_density(params)
        length_scale = 0.5 * (
            example_density.minimum_width + example_density.minimum_spacing
        )
        fwhm = length_scale * params.metadata.length_scale_fwhm_factor
        sigma = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))

        s_factor = params.metadata.smoothing_factor
        highres_i = (
            0.5
            + jnp.arange(
                s_factor * (-pad_pixels),
                s_factor * (pad_pixels + example_density.shape[-2]),
            )
        ) / s_factor
        highres_j = (
            0.5
            + jnp.arange(
                s_factor * (-pad_pixels),
                s_factor * (pad_pixels + example_density.shape[-1]),
            )
        ) / s_factor

        # Coordinates for the control points of the Gaussian radial basis functions.
        levelset_i, levelset_j = _control_point_coords(
            density_shape=params.metadata.density_shape[-2:],  # type: ignore[arg-type]
            levelset_shape=(
                params.latents.amplitude.shape[-2:]  # type: ignore[arg-type]
            ),
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

        levelset_i = levelset_i.flatten()
        levelset_j = levelset_j.flatten()

    amplitude = params.latents.amplitude
    if example_density.periodic[0]:
        amplitude = jnp.concat([amplitude] * 3, axis=-2)
    if example_density.periodic[1]:
        amplitude = jnp.concat([amplitude] * 3, axis=-1)

    amplitude = amplitude.reshape(amplitude.shape[:-2] + (1, -1))

    # Use a scan operation to compute the array; this lowers memory consumption.
    def scan_fn(_: Tuple[()], i: jnp.ndarray) -> Tuple[Tuple[()], jnp.ndarray]:
        distance_sq = (i - levelset_i) ** 2 + (
            highres_j[:, jnp.newaxis] - levelset_j
        ) ** 2
        basis = jnp.exp(-distance_sq / sigma**2)
        return (), jnp.sum(basis * amplitude, axis=-1)

    _, array = jax.lax.scan(scan_fn, (), xs=highres_i)
    array = jnp.moveaxis(array, 0, -2)

    assert array.shape[-2] % s_factor == 0
    assert array.shape[-1] % s_factor == 0
    array = symmetry.symmetrize(array, tuple(example_density.symmetries))
    return array


# -----------------------------------------------------------------------------
# Functions to compute constraints.
# -----------------------------------------------------------------------------


def analytical_constraints(
    params: GaussianLevelsetParams,
    length_scale_constraint_factor: float,
    length_scale_constraint_beta: float,
    length_scale_constraint_weight: float,
    curvature_constraint_weight: float,
    fixed_pixel_constraint_weight: float,
    mask_gradient: bool,
    pad_pixels: int,
) -> jnp.ndarray:
    """Computes analytical levelset constraints associated with the params."""
    length_scale_constraint, curvature_constraint = _levelset_constraints(
        params,
        beta=length_scale_constraint_beta,
        length_scale_constraint_factor=length_scale_constraint_factor,
        pad_pixels=pad_pixels,
    )
    fixed_pixel_constraint = _fixed_pixel_constraint(
        params,
        mask_gradient=mask_gradient,
        pad_pixels=pad_pixels,
    )

    constraints = jnp.stack(
        [
            length_scale_constraint * length_scale_constraint_weight,
            curvature_constraint * curvature_constraint_weight,
            fixed_pixel_constraint * fixed_pixel_constraint_weight,
        ],
        axis=-1,
    )

    # Normalize constraints to make them (somewhat) resolution-independent.
    example_density = _example_density(params)
    length_scale = 0.5 * (
        example_density.minimum_spacing + example_density.minimum_width
    )
    return constraints / length_scale**2


def _fixed_pixel_constraint(
    params: GaussianLevelsetParams,
    mask_gradient: bool,
    pad_pixels: int,
) -> jnp.ndarray:
    """Return the fixed pixel constraint array.

    The fixed pixel constraint array is nonzero at locations where the density obtained
    from `params` differs from fixed pixels.

    Args:
        params: The parameters from which the density is obtained.
        mask_gradient: If `True`, the gradient is masked so that it is nonzero only at
            the borders of features.
        pad_pixels: The number of pixels added at borders. Values greater than zero
            help to ensure that sharp features at the borders are avoided.

    Returns:
        The constraints array.
    """
    array = _to_array(params, mask_gradient=mask_gradient, pad_pixels=pad_pixels)

    example_density = _example_density(params)
    fixed_solid = jnp.zeros(example_density.shape[-2:], dtype=bool)
    fixed_void = jnp.zeros(example_density.shape[-2:], dtype=bool)
    if example_density.fixed_solid is not None:
        fixed_solid = jnp.asarray(example_density.fixed_solid)
    if example_density.fixed_void is not None:
        fixed_void = jnp.asarray(example_density.fixed_void)

    pad_width_solid = ((0, 0),) * (fixed_solid.ndim - 2) + (
        (pad_pixels, pad_pixels),
        (pad_pixels, pad_pixels),
    )
    pad_width_void = ((0, 0),) * (fixed_void.ndim - 2) + (
        (pad_pixels, pad_pixels),
        (pad_pixels, pad_pixels),
    )
    fixed_solid = jnp.pad(fixed_solid, pad_width_solid, mode="edge")
    fixed_void = jnp.pad(fixed_void, pad_width_void, mode="edge")
    fixed = fixed_solid | fixed_void
    target = jnp.where(fixed_solid, 1, 0)

    return jnp.where(fixed, jnp.abs(array - target), 0.0)


def _levelset_constraints(
    params: GaussianLevelsetParams,
    beta: float,
    length_scale_constraint_factor: float,
    pad_pixels: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute constraints for minimum width, spacing, and radius of curvature.

    The constraints are based on "Analytical level set fabrication constraints for
    inverse design," by D. Vercruysse et al. (2019). Constraints are satisfied when
    they are non-positive.

    https://www.nature.com/articles/s41598-019-45026-0

    Args:
        params: The parameters of the Gaussian levelset.
        beta: Parameter which relaxes the constraint near the zero-plane.
        length_scale_constraint_factor: Multiplies the target length scale in the
            levelset constraints. A value greater than 1 is pessimistic and drives the
            solution to have a larger length scale (relative to smaller values).
        pad_pixels: A non-negative integer giving the additional pixels to be included
            beyond the boundaries of the parameterized density.

    Returns:
        The minimum length scale and minimum curvature constraint arrays.s
    """
    example_density = _example_density(params)
    minimum_length_scale = 0.5 * (
        example_density.minimum_width + example_density.minimum_spacing
    )

    phi, phi_v, phi_vv, inverse_radius = _phi_derivatives_and_inverse_radius(
        params,
        pad_pixels=pad_pixels,
    )

    d = minimum_length_scale * length_scale_constraint_factor
    denom = jnp.pi / d * jnp.abs(phi) + beta * phi_v
    denom_safe = jnp.where(jnp.isclose(phi_vv, 0.0), 1.0, denom)
    length_scale_constraint = jnp.abs(phi_vv) / denom_safe - jnp.pi / d

    curvature_denom_safe = jnp.where(jnp.isclose(phi_v, 0.0), 1.0, phi)
    curvature_constraint = (
        jnp.abs(inverse_radius * jnp.arctan(phi_v / curvature_denom_safe)) - jnp.pi / d
    )

    # Downsample so that constraints shape matches the density shape.
    factor = params.metadata.smoothing_factor
    return (
        _downsample_spatial_dims(length_scale_constraint, factor),
        _downsample_spatial_dims(curvature_constraint, factor),
    )


def _phi_derivatives_and_inverse_radius(
    params: GaussianLevelsetParams,
    pad_pixels: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the levelset function and its first and second derivatives."""

    phi = _phi_from_params(
        params=params,
        pad_pixels=pad_pixels,
    )

    d = 1 / params.metadata.smoothing_factor
    phi_x, phi_y = jnp.gradient(phi, d, axis=(-2, -1))
    phi_xx, phi_yx = jnp.gradient(phi_x, d, axis=(-2, -1))
    phi_xy, phi_yy = jnp.gradient(phi_y, d, axis=(-2, -1))

    phi_v = _sqrt_safe(phi_x**2 + phi_y**2)

    # Compute "safe" versions of `phi_v` and its square, which are used to
    # normalize quantities below. These are equal to 1 anywhere `phi_v` is
    # close to zero, and take their usual values elsewhere.
    phi_v_near_zero = jnp.isclose(phi_v, 0.0)
    phi_v_squared_safe = jnp.where(phi_v_near_zero, 1.0, phi_v**2)
    phi_v_safe = jnp.where(phi_v_near_zero, 1.0, phi_v)

    weight_xx = phi_x**2 / phi_v_squared_safe
    weight_yy = phi_y**2 / phi_v_squared_safe
    weight_xy = (phi_x * phi_y) / phi_v_squared_safe
    phi_vv = weight_xx * phi_xx + weight_xy * (phi_xy + phi_yx) + weight_yy * phi_yy

    qx: jnp.ndarray = jnp.gradient(  # type: ignore[assignment]
        phi_x / jnp.abs(phi_v_safe), d, axis=-2
    )
    qy: jnp.ndarray = jnp.gradient(  # type: ignore[assignment]
        phi_y / jnp.abs(phi_v_safe), d, axis=-1
    )
    inverse_radius = qx + qy

    return phi, phi_v, phi_vv, inverse_radius


# -----------------------------------------------------------------------------
# Helper functions.
# -----------------------------------------------------------------------------


def _control_point_coords(
    density_shape: Tuple[int, int],
    levelset_shape: Tuple[int, int],
    periodic: Tuple[bool, bool],
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
    """Compute square root while avoiding `nan` gradients near zero."""
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
        interface = _interface_pixels(phi, periodic)
        phi = jnp.where(interface, phi, jax.lax.stop_gradient(phi))
    thresholded = (phi > 0).astype(float) + (phi - jax.lax.stop_gradient(phi))
    return thresholded


def _interface_pixels(phi: jnp.ndarray, periodic: Tuple[bool, bool]) -> jnp.ndarray:
    """Identifies interface pixels of a level set function `phi`."""
    batch_shape = phi.shape[:-2]
    phi = phi.reshape((-1,) + phi.shape[-2:])

    pad_mode = (
        "wrap" if periodic[0] else "edge",
        "wrap" if periodic[1] else "edge",
    )
    pad_width = ((1, 1), (1, 1))

    kernel = jnp.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)

    solid = phi > 0
    void = ~solid

    solid_padded = transforms.pad2d(solid, pad_width, pad_mode)
    num_solid_adjacent = transforms.conv(
        x=solid_padded[:, jnp.newaxis, :, :].astype(float),
        kernel=kernel[jnp.newaxis, jnp.newaxis, :, :],
        padding="VALID",
    )
    num_solid_adjacent = jnp.squeeze(num_solid_adjacent, axis=1)

    void_padded = transforms.pad2d(void, pad_width, pad_mode)
    num_void_adjacent = transforms.conv(
        x=void_padded[:, jnp.newaxis, :, :].astype(float),
        kernel=kernel[jnp.newaxis, jnp.newaxis, :, :],
        padding="VALID",
    )
    num_void_adjacent = jnp.squeeze(num_void_adjacent, axis=1)

    interface = solid & (num_void_adjacent > 0) | void & (num_solid_adjacent > 0)

    return interface.reshape(batch_shape + interface.shape[-2:])


def _downsample_spatial_dims(x: jnp.ndarray, downsample_factor: int) -> jnp.ndarray:
    """Downsamples the two trailing axes of `x` by `downsample_factor`."""
    shape = x.shape[:-2] + (
        x.shape[-2] // downsample_factor,
        x.shape[-1] // downsample_factor,
    )
    return transforms.box_downsample(x, shape)
