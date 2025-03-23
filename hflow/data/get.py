import os
import time
from functools import partial
from pathlib import Path
import jax
import h5py
import numpy as np
import pandas as pd
from einops import rearrange
from jax import jit, vmap

import hflow.io.result as R
from hflow.config import Data
from hflow.data.lorenz9 import get_ic_lorenz9d, get_lorenz9d
from hflow.data.mdyn import get_mdyn_sol
from hflow.data.particles import (get_2d_bi, get_2d_lin, get_ic_bi,
                                  get_ic_lin)
from hflow.data.sde import solve_sde
from hflow.data.trap import get_ic_trap, get_trap
from hflow.data.utils import normalize
from hflow.data.vlasov import run_vlasov
from hflow.io.utils import log, save_pickle
from hflow.data.ip import get_inertial_partices


def read_from_hdf5(path, n_samples):
    with h5py.File("/scratch/jmb1174/data/" + path + ".hdf5", "r") as file:
        t_grid = file["t_grid"][:]
        sol = file["sol"][:, 0:n_samples, :]
        mu = file["mu"][()]
        time = file["time"][()]
    return sol, mu, t_grid, time


def get_data(problem, data_cfg: Data, key):

    log.info(f'getting data...')

    n_samples = data_cfg.n_samples
    dt, t_end = data_cfg.dt, data_cfg.t_end
    t_eval = np.linspace(0.0, t_end, int(t_end/dt)+1)

    sols = []

    if problem == 'static':
        train_mus = np.asarray([0.0])
        test_mus = np.asarray([0.0])
        mus = np.concatenate([train_mus, test_mus])
        sols = jax.random.normal(key, shape=(2,len(t_eval), n_samples, 2))
        sols = np.asarray(sols)
        
    elif problem == 'ip':
        sols_train, train_mus, t_eval = get_inertial_partices(key, n_samples, train=True)
        sols_test, test_mus, t_eval = get_inertial_partices(key, n_samples, train=False)
        mus = np.concatenate([train_mus, test_mus])
        sols = np.concatenate([sols_train, sols_test])

    elif problem == 'vbump':
        train_mus = np.asarray([1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        test_mus = np.asarray([1.35, 1.95])
        mus = np.concatenate([train_mus, test_mus])
        for mu_i, mu in enumerate(mus):
            start = time.time()
            res = run_vlasov(n_samples, t_eval, mu,
                             mode='bump-on-tail', eta=1e-3)
            end = time.time()
            R.RESULT[f'FOM_integrate_time_{mu_i}'] = end-start
            sols.append(res)
        sols = np.asarray(sols)
    elif problem == 'vtwo':
        train_mus = np.asarray([1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
        test_mus = np.asarray([1.25, 1.85])
        mus = np.concatenate([train_mus, test_mus])
        for mu_i, mu in enumerate(mus):
            start = time.time()
            res = run_vlasov(n_samples, t_eval, mu, mode='two-stream')
            end = time.time()
            R.RESULT[f'FOM_integrate_time_{mu_i}'] = end-start
            sols.append(res)
        sols = np.asarray(sols)
    elif problem == 'lz9':
        train_mus = np.asarray(
            [13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2])
        test_mus = np.asarray([13.65, 14.05])
        mus = np.concatenate([train_mus, test_mus])

        def solve_for_mu(mu):
            drift, diffusion = get_lorenz9d(mu, noise=5e-2)
            return solve_sde(drift, diffusion, t_eval, get_ic_lorenz9d, n_samples, dt=data_cfg.dt, key=key)
        for mu in mus:
            res = solve_for_mu(mu)
            sols.append(res)
        sols = np.asarray(sols)
        sols = rearrange(sols, 'M N T D -> M T N D')

    elif problem == 'bi':

        train_mus = np.asarray([0.10, 0.15, 0.25, 0.3])
        # train_mus = np.asarray([0.2])
        test_mus = np.asarray([0.2])
        mus = np.concatenate([train_mus, test_mus])

        def solve_for_mu(mu):
            drift, diffusion = get_2d_bi(mu)
            return solve_sde(drift, diffusion, t_eval, get_ic_bi, n_samples, dt=data_cfg.dt, key=key)
        sols = vmap(jit(solve_for_mu))(mus)
        sols = rearrange(sols, 'M N T D -> M T N D')

    elif problem == 'lin':
        train_mus = np.asarray([data_cfg.omega])
        test_mus = np.asarray([data_cfg.omega])
        mus = np.concatenate([train_mus, test_mus])

        def solve_for_mu(mu):
            drift, diffusion = get_2d_lin(mu)
            return solve_sde(drift, diffusion, t_eval, get_ic_lin, n_samples, dt=data_cfg.dt, key=key)
        sols = vmap(jit(solve_for_mu))(mus)
        sols = rearrange(sols, 'M N T D -> M T N D')
        sols = sols[:, :, :, :2]
        
    elif problem == 'trap':
        train_mus = np.asarray([0.3, 0.4, 0.5, 0.7, 0.8, 0.9])
        test_mus = np.asarray([0.6])
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

    elif problem == "v6":
        mus = []
        for i in range(0, 11):
            sol, mu, t_grid, wall_time = read_from_hdf5(
                "strongLandauDampingCV" + f"{i:02d}", n_samples)
            R.RESULT[f'FOM_integrate_time_{i}'] = wall_time
            mus.append(mu)
            sols.append(sol)
        sols = np.asarray(sols)
        mus = np.asarray(mus)
        idx = np.argsort(mus)
        mus = mus[idx]
        sols = sols[idx]
        train_idx = np.asarray([0, 1, 2, 3, 4, 6, 7, 9, 10])
        test_idx = np.asarray([5])
        train_mus = mus[train_idx]
        test_mus = mus[test_idx]

        t_eval = t_grid
        sols = sols[:, :-1]  # bug sol is too big
        T = sols.shape[1]//data_cfg.t_end
        T = int(T)

        sols = sols[:, :T]
        t_eval = t_eval[:T]

        
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

    if normalize_data:
        if len(np.unique(mus_1[:, 0])) == 1:
            mus_1 = mus_1*0.0
            (mu_shift, mu_scale) = 0.0, 0.0
        else:
            mus_1, mu_shift, mu_scale = normalize(
                mus_1, axis=(0), return_stats=True, method='std')
    else:
        mu_shift, mu_scale = 0.0, 1.0
    R.RESULT['mu_norm'] = (mu_shift, mu_scale)  

    if normalize_data:
        sols, d_shift, d_scale = normalize(
            sols, axis=(0, 1, 2), return_stats=True, method='01')
        R.RESULT['data_norm'] = (d_shift, d_scale)
    else:
        R.RESULT['data_norm'] = (0.0, 1.0)


    return sols, mus_1, t1
