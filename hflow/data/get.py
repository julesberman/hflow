import numpy as np
from einops import rearrange
from jax import jit, vmap

import hflow.io.result as R
from hflow.config import Train_Data
from hflow.data.particles import get_2d_osc, get_ic_osc
from hflow.data.sde import solve_sde
from hflow.data.utils import normalize
from hflow.data.vlasov import run_vlasov
from hflow.io.utils import log


def get_data(problem, data_cfg: Train_Data, batch_size, key):

    n_samples = data_cfg.n_samples
    dt, t_end = data_cfg.dt, data_cfg.t_end
    t_eval = np.linspace(0.0, t_end, int(t_end/dt)+1)
    mus = np.asarray([0.1, 0.15, 0.2])

    sols = []
    match problem:
        case 'vlasov':
            for mu in mus:
                res = run_vlasov(n_samples, t_eval, mu)
                sols.append(res)
            sols = np.asarray(sols)
        case 'osc':
            def solve_for_mu(mu):
                drift, diffusion = get_2d_osc(mu)
                return solve_sde(drift, diffusion, t_eval, get_ic_osc, n_samples)
            sols = vmap(jit(solve_for_mu))(mus)

    # sub sample time
    T = sols.shape[1]
    nn = T // data_cfg.n_time
    sols = sols[:, ::nn]
    t_eval = t_eval[::nn]
    log.info(f'train data (M x T x N x D) {sols.shape}')

    R.RESULT['t_eval'] = t_eval
    R.RESULT['train_data'] = sols

    sols, mu_t = normalize_dataset(sols, mus, t_eval)
    data = (sols, mu_t)

    data_fn = get_data_fn(sols)

    return data


def get_data_fn(sols, t_data, bs_n, bs_t):
    T, N, D = sols.shape

    def args_fn(key):

        nonlocal sols
        nonlocal t_data

        key, keyt = jax.random.split(key)

        t_idx = jax.random.choice(keyt, jnp.arange(
            1, T-1), shape=(bs_t-2,), replace=False)
        start, end = jnp.asarray([0]), jnp.asarray([T-1])
        t_idx = jnp.concatenate([start, t_idx, end])

        keys = jax.random.split(key, num=bs_t)
        sample_idx = vmap(get_rand_idx, (0, None, None))(keys, N, bs_n)

        sols_sample = sols[t_idx]
        t_sample = t_data[t_idx]

        sols_sample = sols_sample[jnp.arange(len(sols_sample))[
            :, None], sample_idx]

        return sols_sample, t_sample
    return args_fn


def normalize_dataset(sols, mus, t_eval):
    t1 = t_eval.reshape((-1, 1))
    mus_1 = mus.reshape((-1, 1))
    mu_t = np.concatenate(
        [np.repeat(mus_1, len(t1), axis=0), np.tile(t1, (len(mus_1), 1))],
        axis=1
    )

    sols, d_shift, d_scale = normalize(
        sols, axis=(0, 1, 2), return_stats=True, method='01')
    mu_t, mu_t_shift, mu_t_scale = normalize(
        mu_t, axis=(0), return_stats=True, method='std')

    R.RESULT['data_norm'] = (d_shift, d_scale)
    R.RESULT['mu_t_norm'] = (mu_t_shift, mu_t_scale)
    R.RESULT['mu_t_train'] = mu_t

    return sols, mu_t
