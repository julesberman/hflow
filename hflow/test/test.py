import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from jax import jacrev, jit, vmap
from tqdm import tqdm

import hflow.io.result as R
from hflow.config import Config
from hflow.data.ode import odeint_euler_maruyama
from hflow.data.sde import solve_sde_ic
from hflow.io.utils import log
from hflow.misc.jax import batchmap, get_rand_idx
from hflow.test.metrics import compute_metrics
from hflow.test.plot import plot_test
from hflow.train.loss import generate_sigmas


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
        log.info(f'testing mu {mus[mu_i]} {mu_i}')
        true_sol = sol[mu_i]
        if cfg.loss.loss_fn == 'ov':
            test_sol = solve_test_sde(s_fn, opt_params, ics[mu_i], t_int,
                                      test_cfg.dt, sigma, mus[mu_i], key)
        elif cfg.loss.loss_fn == 'ncsm':
            sigmas = generate_sigmas(cfg.loss)
            test_sol = solve_test_ald(
                s_fn, opt_params, ics[mu_i], t_int, sigmas, mus[mu_i], key)

        if test_cfg.save_sol:
            R.RESULT[f'true_sol_{mu_i}'] = true_sol
            R.RESULT[f'test_sol_{mu_i}'] = test_sol

        compute_metrics(test_cfg, true_sol, test_sol, mu_i)

        plot_test(test_cfg, true_sol, test_sol,
                  t_int, test_cfg.n_plot_samples, mu_i)

    return test_sol


def solve_test_sde(s_fn, params, ics, t_int, dt, sigma, mu, key):
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


def solve_test_ald(s_fn, params, ics, t_int, sigmas, mu, key):

    D = ics.shape[-1]
    n_samples = ics.shape[0]

    def ald(key, mu, t_infer, sigmas, params, eps=2e-5, T=10):
        x = jax.random.normal(key, (1, D))

        for sigma in sigmas:
            alpha = eps * sigma**2/sigmas[-1]**2
            for _ in range(T):
                key, subkey = jax.random.split(key)
                z = jax.random.normal(subkey, x.shape)
                mu_t_sigma = jnp.concatenate([mu, t_infer, sigma])
                x = x + alpha * 0.5 * \
                    s_fn(mu_t_sigma, x, params) + jnp.sqrt(alpha) * z
        return x

    ald_vmap = vmap(ald, (0, None, None, None, None))
    ald_vmap = jit(ald_vmap)
    # ald_vmap_time = vmap(ald_vmap, (None, None, 0, None, None))
    # ald_vmap_time = batchmap(ald_vmap_time, 8)

    t_int = t_int.reshape(-1, 1)
    keys = jax.random.split(key, n_samples)
    test_sol = []
    for t in tqdm(t_int):
        s = ald_vmap(keys, mu, t, sigmas, params)
        test_sol.append(s)

    test_sol = jnp.asarray(test_sol)
    test_sol = jnp.squeeze(test_sol)

    return test_sol
