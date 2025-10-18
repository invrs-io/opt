"""Defines tests for the Gaussian levelset parameterization.

Copyright (c) 2025 invrs.io LLC
"""

import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized
from totypes import types

from invrs_opt.parameterization import gaussian_levelset


class GaussianLevelsetTest(unittest.TestCase):
    @parameterized.expand(itertools.product([1, 2, 3], [(0, 1), (-1, 1), (1, 2)]))
    def test_initial_density_matches_expected(self, length_scale, bounds):
        lb, ub = bounds
        array = onp.zeros((10, 12))
        array[:, 4:8] = 1
        array = lb + array * (ub - lb)

        density = types.Density2DArray(
            array=array,
            lower_bound=lb,
            upper_bound=ub,
            minimum_width=length_scale,
            minimum_spacing=length_scale,
        )

        parameterization = gaussian_levelset.gaussian_levelset()
        latent_params = parameterization.from_density(density)
        initial_density = parameterization.to_density(latent_params)

        onp.testing.assert_allclose(initial_density.array, density.array)

    def test_jit(self):
        density = types.Density2DArray(
            array=onp.zeros((10, 12)),
            lower_bound=0,
            upper_bound=1,
            minimum_width=3,
            minimum_spacing=3,
        )

        parameterization = gaussian_levelset.gaussian_levelset()
        latent_params = parameterization.from_density(density)

        @jax.jit
        def jit_fn(latent_params):
            params = parameterization.to_density(latent_params)
            return jnp.sum(params.array)

        jit_fn(latent_params)

    def test_constraints_grad_has_no_nan(self):
        array = onp.zeros((20, 20))
        array[:, :10] = 1

        density = types.Density2DArray(array=array, lower_bound=0, upper_bound=1)

        parameterization = gaussian_levelset.gaussian_levelset()
        latent_params = parameterization.from_density(density)

        latents = latent_params.latents
        metadata = latent_params.metadata

        def _loss_fn(latents):
            latent_params = gaussian_levelset.GaussianLevelsetParams(
                latents=latents, metadata=metadata
            )
            constraints = parameterization.constraints(latent_params)
            return jnp.sum(constraints**2)

        grad = jax.grad(_loss_fn)(latents)
        self.assertFalse(onp.any(onp.isnan(grad.amplitude)))
