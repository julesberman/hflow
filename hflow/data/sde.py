import random
from functools import partial

import jax.numpy as jnp
import jax.random
from diffrax import (Euler, Heun, MultiTerm, ODETerm, SaveAt,
                     VirtualBrownianTree, WeaklyDiagonalControlTerm,
                     diffeqsolve)
from jax import jit, vmap

from hflow.misc.jax import batchmap


def solve_sde(drift, diffusion, t_eval, get_ic, n_samples, dt=1e-2, key=None, batches=1):
    t_eval = jnp.asarray(t_eval)

    @jit
    def solve_single(key):
        ikey, skey = jax.random.split(key)
        y0 = get_ic(ikey)
        sol = solve_sde_ic(y0, skey, t_eval, dt, drift, diffusion)
        return sol

    if key is None:
        key = jax.random.PRNGKey(random.randint(0, 1e6))
    keys = jax.random.split(key, num=n_samples)
    solve_single = vmap(solve_single)
    if batches > 1:
        solve_single = batchmap(solve_single, batches)

    sols = solve_single(keys)

    return sols


def solve_sde_ic(y0, key, t_eval, dt, drift, diffusion):
    t0, t1 = t_eval[0], t_eval[-1]
    brownian_motion = VirtualBrownianTree(
        t0, t1, tol=1e-3, shape=y0.shape, key=key)
    terms = MultiTerm(ODETerm(drift), WeaklyDiagonalControlTerm(
        diffusion, brownian_motion))
    solver = Euler()
    saveat = SaveAt(ts=t_eval)
    sol = diffeqsolve(terms, solver, t0, t1, dt0=dt,
                      y0=y0, saveat=saveat, max_steps=int(1e6))

    return sol.ys
