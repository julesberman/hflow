import random

import jax.random
from diffrax import (ControlTerm, Euler, Heun, MultiTerm, ODETerm, SaveAt,
                     VirtualBrownianTree, diffeqsolve)
from jax import jit, vmap


def solve_sde(drift, diffusion, t_eval, get_ic, n_samples, dt=1e-2, key=None):

    @jit
    def solve_single(key):
        t0, t1 = t_eval[0], t_eval[-1]
        ikey, skey = jax.random.split(key)
        y0 = get_ic(ikey)
        brownian_motion = VirtualBrownianTree(
            t0, t1, tol=1e-3, shape=(), key=skey)
        terms = MultiTerm(ODETerm(drift), ControlTerm(
            diffusion, brownian_motion))
        solver = Euler()
        saveat = SaveAt(ts=t_eval)
        sol = diffeqsolve(terms, solver, t0, t1, dt0=dt,
                          y0=y0, saveat=saveat, max_steps=int(1e6))
        return sol.ys

    if key is None:
        key = jax.random.PRNGKey(random.randint(0, 1e6))
    keys = jax.random.split(key, num=n_samples)
    sols = vmap(solve_single)(keys)

    return sols
