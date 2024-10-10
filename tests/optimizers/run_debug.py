"""Defines tests for the `lbfgsb.lbfgsb` module.

Copyright (c) 2023 The INVRS-IO authors.
"""

import argparse

import jax
import jax.numpy as jnp
from totypes import types

from invrs_opt.optimizers import lbfgsb

jax.config.update("jax_enable_x64", True)


def optimization_with_vmap(steps):
    params = types.Density2DArray(array=jnp.ones((2, 3, 3)))
    opt = lbfgsb.density_lbfgsb(beta=2)
    state = jax.vmap(opt.init)(params)

    @jax.jit
    @jax.vmap
    def step_fn(state):
        params = opt.params(state)
        dummy_value = jnp.array(1.0, dtype=float)
        dummy_grad = jax.tree_util.tree_map(jnp.ones_like, params)
        state = opt.update(
            grad=dummy_grad, value=dummy_value, params=params, state=state
        )
        return state, dummy_value

    for i in range(steps):
        print(f"vmap step {i}", flush=True)
        state, value = step_fn(state)


parser = argparse.ArgumentParser(prog="debug", description="opt debugging")
parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help="Number of steps.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    optimization_with_vmap(steps=args.steps)
