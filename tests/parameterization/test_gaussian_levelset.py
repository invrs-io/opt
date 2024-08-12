"""Defines tests for the Gaussian levelset parameterization.

Copyright (c) 2023 The INVRS-IO authors.
"""

import itertools
import unittest

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
