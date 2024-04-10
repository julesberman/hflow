import numpy as np
from einops import rearrange
from jax import jit, vmap

import hflow.io.result as R
from hflow.config import Data
from hflow.data.particles import get_2d_osc, get_ic_osc
from hflow.data.sde import solve_sde
from hflow.data.utils import normalize
from hflow.data.vlasov import run_vlasov
from hflow.io.utils import log
from hflow.truth.sburgers import solve_sburgers_samples


def get_data(problem, data_cfg: Data, key):

    n_samples = data_cfg.n_samples
    dt, t_end = data_cfg.dt, data_cfg.t_end
    t_eval = np.linspace(0.0, t_end, int(t_end/dt)+1)

    sols = []

    if problem == 'vlasov':
        mus = np.asarray([0.1, 0.15, 0.2])
        for mu in mus:
            res = run_vlasov(n_samples//2, t_eval, mu)
            sols.append(res)
        sols = np.asarray(sols)
    elif problem == 'osc':
        mus = np.asarray([0.15, 0.1, 0.05])

        def solve_for_mu(mu):
            drift, diffusion = get_2d_osc(mu)
            return solve_sde(drift, diffusion, t_eval, get_ic_osc, n_samples, dt=data_cfg.dt, key=key)
        sols = vmap(jit(solve_for_mu))(mus)
        sols = rearrange(sols, 'M N T D -> M T N D')
    elif problem == 'sburgers':
        mus = np.asarray([1e-3, 5e-3, 1e-2])
        N = 256
        sub_N = 4
        sigma = 1e-1
        modes = 100
        sols = solve_sburgers_samples(
            n_samples, mus, N, sub_N, sigma, modes, t_eval, key)
        sols = rearrange(sols, 'M N T D -> M T N D')

    log.info(f'train data (M x T x N x D) {sols.shape}')

    R.RESULT['t_eval'] = t_eval
    R.RESULT['sols'] = sols

    sols, mu, t = normalize_dataset(sols, mus, t_eval, data_cfg.normalize)
    data = (sols, mu, t)

    return data


def normalize_dataset(sols, mus, t_eval, normalize_data):
    t1 = t_eval.reshape((-1, 1))
    mus_1 = mus.reshape((-1, 1))

    mus_1, mu_shift, mu_scale = normalize(
        mus_1, axis=(0), return_stats=True, method='std')

    if normalize_data:
        sols, d_shift, d_scale = normalize(
            sols, axis=(0, 1, 2), return_stats=True, method='01')
        R.RESULT['data_norm'] = (d_shift, d_scale)

    R.RESULT['mu_norm'] = (mu_shift, mu_scale)
    R.RESULT['mu_train'] = mus_1

    return sols, mus_1, t1
