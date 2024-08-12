"""Defines tests across all density parameterizations.

Copyright (c) 2023 The INVRS-IO authors.
"""

import itertools
import unittest

import jax.numpy as jnp
import numpy as onp
from jax import tree_util
from parameterized import parameterized
from totypes import json_utils, types

from invrs_opt.parameterization import gaussian_levelset, filter_project, pixel


PARAMETERIZATIONS = (
    gaussian_levelset.gaussian_levelset(),
    filter_project.filter_project(beta=2.0),
    pixel.pixel(),
)

# Define various densities with and without batch dimensions, and with and without
# fixed solid and void pixels.
DENSITIES = (
    types.Density2DArray(
        array=jnp.zeros((10, 12)),
        fixed_solid=None,
        fixed_void=None,
    ),
    types.Density2DArray(
        array=jnp.zeros((3, 10, 12)),
        fixed_solid=None,
        fixed_void=None,
    ),
    types.Density2DArray(
        array=jnp.zeros((10, 12)),
        fixed_solid=jnp.zeros((10, 12), dtype=bool).at[4:8, 0].set(True),
        fixed_void=None,
    ),
    types.Density2DArray(
        array=jnp.zeros((3, 10, 12)),
        fixed_solid=jnp.zeros((3, 10, 12), dtype=bool).at[:, 4:8, 0].set(True),
        fixed_void=None,
    ),
    types.Density2DArray(
        array=jnp.zeros((10, 12)),
        fixed_solid=jnp.zeros((10, 12), dtype=bool).at[4:8, 0].set(True),
        fixed_void=jnp.zeros((10, 12), dtype=bool).at[4:8, -1].set(True),
    ),
    types.Density2DArray(
        array=jnp.zeros((3, 10, 12)),
        fixed_solid=jnp.zeros((3, 10, 12), dtype=bool).at[:, 4:8, 0].set(True),
        fixed_void=jnp.zeros((3, 10, 12), dtype=bool).at[:, 4:8, -1].set(True),
    ),
)


class ParameterizationTest(unittest.TestCase):
    @parameterized.expand(itertools.product(PARAMETERIZATIONS, DENSITIES))
    def test_convert_from_and_to_density(self, parameterization, density):
        latent_params = parameterization.from_density(density)
        restored_density = parameterization.to_density(latent_params)
        self.assertEqual(
            tree_util.tree_structure(density),
            tree_util.tree_structure(restored_density),
        )
        self.assertSequenceEqual(density.shape, restored_density.shape)

    @parameterized.expand(itertools.product(PARAMETERIZATIONS, DENSITIES))
    def test_compute_constraints(self, parameterization, density):
        latent_params = parameterization.from_density(density)
        _ = parameterization.constraints(latent_params)

    @parameterized.expand(itertools.product(PARAMETERIZATIONS, DENSITIES))
    def test_latent_params_are_serializable(self, parameterization, density):
        latent_params = parameterization.from_density(density)
        serialized = json_utils.json_from_pytree(latent_params)
        deserialized = json_utils.pytree_from_json(serialized)
        self.assertEqual(
            tree_util.tree_structure(deserialized),
            tree_util.tree_structure(latent_params),
        )
        for a, b in zip(
            tree_util.tree_leaves(latent_params),
            tree_util.tree_leaves(deserialized),
        ):
            onp.testing.assert_array_equal(a, b)
        self.assertEqual(type(deserialized), type(latent_params))

    @parameterized.expand(
        itertools.product(PARAMETERIZATIONS, [(0, 1), (-1, 1), (1, 2)])
    )
    def test_density_has_expected_value(self, parameterization, bounds):
        lb, ub = bounds
        array = onp.zeros((20, 20))
        array[:, :10] = 1
        array = lb + array * (ub - lb)

        density = types.Density2DArray(array=array, lower_bound=lb, upper_bound=ub)

        latent_params = parameterization.from_density(density)
        restored_density = parameterization.to_density(latent_params)

        onp.testing.assert_allclose(onp.amax(restored_density.array), ub)
        onp.testing.assert_allclose(onp.amin(restored_density.array), lb)
