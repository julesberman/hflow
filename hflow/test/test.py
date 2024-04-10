import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from jax import jacrev, jit, vmap

import hflow.io.result as R
from hflow.config import Config
from hflow.data.sde import solve_sde_ic
from hflow.io.utils import log
from hflow.misc.jax import get_rand_idx
from hflow.test.metrics import compute_metrics
from hflow.test.plot import plot_test


def test_model(cfg: Config, data, s_fn, opt_params, key):
    test_cfg = cfg.test
    if not test_cfg.run:
        return None
    sol, mus, t = data
    t_int = np.linspace(0.0, 1.0, len(t))
    M, T, N, D = sol.shape
    samples_idx = get_rand_idx(key, sol.shape[2], test_cfg.n_samples)
    ics = sol[:, 0, samples_idx, :]

    sigma = cfg.loss.sigma
    sol = sol[:, :, samples_idx]

    R.RESULT['t_int'] = t_int
    for mu_i in range(len(mus)):
        true_sol = sol[mu_i]
        test_sol = solve_test(s_fn, opt_params, ics[mu_i], t_int,
                              test_cfg.dt, sigma, mus[mu_i], key)
        R.RESULT[f'true_sol_{mu_i}'] = true_sol
        R.RESULT[f'test_sol_{mu_i}'] = test_sol

        compute_metrics(test_cfg, true_sol, test_sol, mu_i)

        plot_test(test_cfg, true_sol, test_sol,
                  t_int, test_cfg.plot_samples, mu_i)

    return test_sol


def solve_test(s_fn, params, ics, t_int, dt, sigma, mu, key):
    s_dx = jacrev(s_fn, 1)

    def drift(t, y, *args):
        mu_t = jnp.concatenate([mu, t.reshape(1)])
        f = jnp.squeeze(s_dx(mu_t, y, params))
        return f

    def diffusion(t, y, *args):
        return sigma * jnp.ones_like(y)

    keys = jax.random.split(key, num=len(ics))
    test_sol = vmap(solve_sde_ic, (0, 0, None, None, None, None))(
        ics, keys, t_int, dt, drift, diffusion)
    test_sol = rearrange(test_sol, 'N T D -> T N D')

    return test_sol
