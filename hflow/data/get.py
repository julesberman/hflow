import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from einops import rearrange
from jax import jit, vmap

import hflow.io.result as R
from hflow.config import Data
from hflow.data.lorenz9 import get_ic_lorenz9d, get_lorenz9d
from hflow.data.mdyn import get_mdyn_sol
from hflow.data.particles import (get_2d_bi, get_2d_lin, get_2d_van, get_ic_bi,
                                  get_ic_lin, get_ic_van)
from hflow.data.sde import solve_sde
from hflow.data.trap import get_ic_trap, get_trap
from hflow.data.utils import normalize
from hflow.data.vlasov import run_vlasov
from hflow.io.utils import log, save_pickle
from hflow.truth.sburgers import solve_sburgers_samples


def get_data(problem, data_cfg: Data, key):

    log.info(f'getting data...')

    n_samples = data_cfg.n_samples
    dt, t_end = data_cfg.dt, data_cfg.t_end
    t_eval = np.linspace(0.0, t_end, int(t_end/dt)+1)

    sols = []

    if problem == 'vbump':
        train_mus = np.asarray([1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
        test_mus = np.asarray([1.35, 1.55, 1.75])
        mus = np.concatenate([train_mus, test_mus])
        for mu in mus:
            res = run_vlasov(n_samples, t_eval, mu,
                             mode='bump-on-tail', eta=1e-3)
            sols.append(res)
        sols = np.asarray(sols)
    elif problem == 'vsingle':
        train_mus = np.asarray([1.3])
        test_mus = np.asarray([1.3])
        mus = np.concatenate([train_mus, test_mus])
        for mu in mus:
            res = run_vlasov(n_samples, t_eval, mu, mode='two-stream')
            sols.append(res)
        sols = np.asarray(sols)
    elif problem == 'vtwo':
        # train_mus = np.asarray([1.6, 1.65, 1.7, 1.75, 1.8])
        # test_mus = np.asarray([1.675, 1.70001, 1.725])

        train_mus = np.asarray([1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
        test_mus = np.asarray([1.35, 1.55, 1.75])
        mus = np.concatenate([train_mus, test_mus])
        for mu in mus:
            res = run_vlasov(n_samples, t_eval, mu, mode='two-stream')
            sols.append(res)
        sols = np.asarray(sols)
    elif problem == 'lz9':

        train_mus = np.asarray([12.1, 12.2, 12.3, 12.5, 12.6, 12.7])
        test_mus = np.asarray([12.4])
        mus = np.concatenate([train_mus, test_mus])

        def solve_for_mu(mu):
            drift, diffusion = get_lorenz9d(mu)
            return solve_sde(drift, diffusion, t_eval, get_ic_lorenz9d, n_samples, dt=data_cfg.dt, key=key)
        for mu in mus:
            res = solve_for_mu(mu)
            sols.append(res)
        sols = np.asarray(sols)
        sols = rearrange(sols, 'M N T D -> M T N D')

    elif problem == 'bi':

        train_mus = np.asarray([0.10, 0.20, 0.3])
        test_mus = np.asarray([0.15, 0.25])
        mus = np.concatenate([train_mus, test_mus])

        def solve_for_mu(mu):
            drift, diffusion = get_2d_bi(mu)
            return solve_sde(drift, diffusion, t_eval, get_ic_bi, n_samples, dt=data_cfg.dt, key=key)
        sols = vmap(jit(solve_for_mu))(mus)
        sols = rearrange(sols, 'M N T D -> M T N D')

    elif problem == 'lin':
        mus = np.asarray([0.10, 0.05, 0.0])

        def solve_for_mu(mu):
            drift, diffusion = get_2d_lin(mu)
            return solve_sde(drift, diffusion, t_eval, get_ic_lin, n_samples, dt=data_cfg.dt, key=key)
        sols = vmap(jit(solve_for_mu))(mus)
        sols = rearrange(sols, 'M N T D -> M T N D')
    elif problem == 'van':
        mus = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0])

        def solve_for_mu(mu):
            drift, diffusion = get_2d_van(mu)
            return solve_sde(drift, diffusion, t_eval, get_ic_van, n_samples, dt=data_cfg.dt, key=key)
        sols = vmap(jit(solve_for_mu))(mus)
        sols = rearrange(sols, 'M N T D -> M T N D')
    elif problem == 'trap':
        train_mus = np.asarray([0.3, 0.4, 0.5, 0.7, 0.8, 0.9])
        test_mus = np.asarray([0.6])

        # train_mus = np.asarray([0.2, 0.8, 1.4, 2.0])
        # test_mus = np.asarray([0.5, 1.1, 1.7])
        mus = np.concatenate([train_mus, test_mus])
        system_dim = data_cfg.dim

        trap = partial(get_trap, system_dim)
        trap_ic = partial(get_ic_trap, system_dim)

        def solve_for_mu(mu):
            drift, diffusion = trap(mu)
            return solve_sde(drift, diffusion, t_eval, trap_ic, n_samples, dt=data_cfg.dt, key=key)
        for mu in mus:
            res = solve_for_mu(mu)
            sols.append(res)
        sols = np.asarray(sols)
        sols = rearrange(sols, 'M N T D -> M T N D')

    elif problem == 'mdyn':
        train_mus = np.asarray([1.0, 3.0, 4.0, 6.0])
        test_mus = np.asarray([2.0, 5.0])
        mus = np.concatenate([train_mus, test_mus])
        system_dim = data_cfg.dim

        mus = np.concatenate([train_mus, test_mus])
        for mu in mus:
            res = get_mdyn_sol(key, system_dim, n_samples, mu,
                               gamma=0.0, alpha=0.0, sigma=1e-1, dt=data_cfg.dt)
            sols.append(res)
        sols = np.asarray(sols)

    R.RESULT['train_mus_raw'] = train_mus
    R.RESULT['test_mus_raw'] = test_mus

    log.info(f'train data (M x T x N x D) {sols.shape}')

    idx = np.argsort(mus)
    mus = mus[idx]
    sols = sols[idx]

    # split train test
    test_indices = np.where(np.isin(mus, test_mus))[0]
    train_indices = np.where(np.isin(mus, train_mus))[0]

    log.info(f'mus: {np.squeeze(mus)}')

    R.RESULT['t_eval'] = t_eval

    sols, mus, t_eval = normalize_dataset(
        sols, mus, t_eval, data_cfg.normalize)

    if data_cfg.save:
        save_data(problem, sols, np.squeeze(mus), np.squeeze(t_eval))
        quit()

    train_sols, test_sols = sols[train_indices], sols[test_indices]
    train_mus, test_mus = mus[train_indices], mus[test_indices]

    R.RESULT['train_mus'] = train_mus
    R.RESULT['test_mus'] = test_mus

    log.info(f'mus_norm: {np.round(np.squeeze(mus),4)}')

    # R.RESULT['train_sols'] = train_sols
    # R.RESULT['test_sols'] = test_sols

    train_data = (train_sols, train_mus, t_eval)
    test_data = (test_sols, test_mus, t_eval)

    return train_data, test_data


def save_data(problem, sols, mu, t):
    sol32 = np.asarray(sols, dtype=np.float32)
    t32 = np.asarray(t, dtype=np.float32)
    mu32 = np.asarray(mu, dtype=np.float32)
    data = {'sols': sol32, 'mu': mu32, 't': t32}
    wd = Path(os.getcwd())
    output_dir = wd / 'ground_truth' / 'sde'
    output_dir.mkdir(exist_ok=True, parents=True)
    output_dir = (output_dir / problem).with_suffix(".pkl")
    save_pickle(output_dir, data)


def load_data(problem):
    wd = Path(os.getcwd())
    output_dir = wd / 'ground_truth' / 'sde'
    output_dir = (output_dir / problem).with_suffix(".pkl")
    log.info(f"loading data from {output_dir}")
    data = pd.read_pickle(output_dir)
    return data


def normalize_dataset(sols, mus, t_eval, normalize_data):
    t1 = t_eval.reshape((-1, 1))
    mus_1 = mus.reshape((-1, 1))

    if len(np.unique(mus_1[:, 0])) == 1:
        mus_1 = mus_1*0.0
        (mu_shift, mu_scale) = 0.0, 0.0
    else:
        mus_1, mu_shift, mu_scale = normalize(
            mus_1, axis=(0), return_stats=True, method='std')

    if normalize_data:
        sols, d_shift, d_scale = normalize(
            sols, axis=(0, 1, 2), return_stats=True, method='01')
        R.RESULT['data_norm'] = (d_shift, d_scale)

    R.RESULT['mu_norm'] = (mu_shift, mu_scale)

    return sols, mus_1, t1
