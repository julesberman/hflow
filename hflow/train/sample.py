from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import vmap
from jax.experimental.host_callback import id_print
from scipy.interpolate import RegularGridInterpolator

import hflow.io.result as R
from hflow.config import Sample
from hflow.io.utils import log
from hflow.misc.jax import get_rand_idx
from hflow.misc.misc import (gauss_quadrature_weights_points,
                             pts_array_from_space)

def get_simpsons(N):
    if (N - 1) % 2 != 0:
            raise ValueError("Number of intervals N - 1 must be even for Simpson's rule (N must be odd).")

    # Generate N equispaced points on [0,1]
    points = np.linspace(0, 1, N)
    h = 1 / (N - 1)  # Step size

    # Initialize weights
    weights = np.zeros(N)
    weights[0] = h / 3
    weights[-1] = h / 3

    # Apply Simpson's coefficients
    for i in range(1, N - 1):
        if i % 2 == 0:
            weights[i] = 2 * h / 3
        else:
            weights[i] = 4 * h / 3

    points = jnp.asarray(points)
    weights = jnp.asarray(weights)

    return points, weights


def get_simpsons_38(N):

    if (N - 1) % 3 != 0:
        raise ValueError("Number of intervals N - 1 must be a multiple of 3 for Simpson's 3/8 rule.")

    points = np.linspace(0, 1, N)
    h = 1 / (N - 1)  # Step size

    # Initialize weights
    weights = np.zeros(N)
    num_groups = (N - 1) // 3

    for group in range(num_groups):
        idx_start = group * 3
        idx_end = idx_start + 3  # Include idx_end in the group

        # Assign weights according to Simpson's 3/8 rule
        weights[idx_start] += 3 * h / 8
        weights[idx_start + 1] += 9 * h / 8
        weights[idx_start + 2] += 9 * h / 8
        weights[idx_start + 3] += 3 * h / 8

    points = jnp.asarray(points)
    weights = jnp.asarray(weights)

    return points, weights


def get_arg_fn(sample_cfg: Sample, data):
    log.info("gettings samples...")
    sols, mu_data, t_data = data
    bs_t = sample_cfg.bs_t
    bs_n = sample_cfg.bs_n
    quad_weights = None
    M, T, N, D = sols.shape
    t_data = t_data / t_data[-1]
    
    w_scheme = sample_cfg.scheme_w

    if sample_cfg.scheme_t == 'fixed':
        sols = jnp.asarray(sols)
        mu_data = jnp.asarray(mu_data)
        t_data = jnp.asarray(t_data)

        def args_fn(key, percent):

            nonlocal sols
            nonlocal t_data
            nonlocal mu_data
       
            sols = jnp.asarray(sols)
            keyn, keym = jax.random.split(key)

            m_idx = jax.random.randint(keym, minval=0, maxval=M, shape=())

            return sols[m_idx], mu_data[m_idx], t_data, quad_weights

        return args_fn
    
    if sample_cfg.scheme_t == 'gauss':
        g_pts_01, quad_weights = gauss_quadrature_weights_points(
            bs_t, a=0.0, b=1.0)
        start, end = jnp.asarray([0]), jnp.asarray([1.0])
        times =  g_pts_01
        g_pts_01 = jnp.concatenate([start, g_pts_01, end])

    elif sample_cfg.scheme_t == 'equi':
        g_pts_01 = jnp.linspace(0.0, 1.0, bs_t)
        start, end = jnp.asarray([0]), jnp.asarray([1.0])
        g_pts_01 = jnp.concatenate([start, g_pts_01, end])

    elif sample_cfg.scheme_t == 'trap':
        g_pts_01 = jnp.linspace(0.0, 1.0, bs_t)
        weight = jnp.concatenate([jnp.asarray([1.0]), 2*jnp.ones((bs_t-2,)), jnp.asarray([1.0])])
        quad_weights = weight * g_pts_01[1] * 0.5
        start, end = jnp.asarray([0]), jnp.asarray([1.0])
        g_pts_01 = jnp.concatenate([start, g_pts_01, end])

    elif sample_cfg.scheme_t == 'simp':
        if (bs_t - 1) % 2 != 0:
            bs_t += 1
        g_pts_01, quad_weights = get_simpsons(bs_t)
        start, end = jnp.asarray([0]), jnp.asarray([1.0])
        g_pts_01 = jnp.concatenate([start, g_pts_01, end])

    elif sample_cfg.scheme_t == 'simp38':
        if (bs_t - 1) % 3 != 0:
            bs_t = 3 * round(bs_t / 3)
        g_pts_01, quad_weights = get_simpsons_38(bs_t)
        start, end = jnp.asarray([0]), jnp.asarray([1.0])
        g_pts_01 = jnp.concatenate([start, g_pts_01, end])



    if sample_cfg.scheme_t != 'rand' and sample_cfg.scheme_t != 'arcsin':
        new_sols = []
        for i, sol_mu in enumerate(sols):
            ss = interplate_in_t(sol_mu, t_data, g_pts_01)
            new_sols.append(ss)

        sols = np.asarray(new_sols)
        t_data = g_pts_01

    # if sample_cfg.scheme_t == 'arcsin':
    #     w_scheme = 'dist'

    log.info(f'sample shape {sols.shape}')
    args_fn = get_data_fn(sols, mu_data, t_data, quad_weights,
                          bs_n, bs_t, sample_cfg.scheme_t, sample_cfg.scheme_n, w_scheme)

    return args_fn


def get_data_fn(sols, mu_data, t_data, quad_weights, bs_n, bs_t, scheme_t, scheme_n, scheme_w):
    M, T, N, D = sols.shape
    t_data = t_data.reshape(-1, 1)

    sols = jnp.asarray(sols)
    mu_data = jnp.asarray(mu_data)
    t_data = jnp.asarray(t_data)

    def args_fn(key, percent):

        nonlocal sols
        nonlocal t_data
        nonlocal mu_data
        nonlocal quad_weights

        keyn, keym, keyt = jax.random.split(key, num=3)

        m_idx = jax.random.randint(keym, minval=0, maxval=M, shape=())
        mu_sample = mu_data[m_idx]
        sols_sample = sols[m_idx]

        if scheme_t == 'rand':
            t_idx = jax.random.choice(keyt, T-1, shape=(bs_t,), replace=False)
            start, end = jnp.asarray([0]), jnp.asarray([T-1])
            t_idx = jnp.concatenate([start, t_idx, end])
            t_idx = jnp.sort(t_idx)
            t_sample = t_data[t_idx]
            sols_sample = sols_sample[t_idx]
        elif scheme_t == 'arcsin':
            t_idx = arcsin_indices(key, bs_t, T-1)
            start, end = jnp.asarray([0]), jnp.asarray([T-1])
            t_idx = jnp.concatenate([start, t_idx, end])
            t_idx = jnp.sort(t_idx)
            t_sample = t_data[t_idx]
            sols_sample = sols_sample[t_idx]
        else:
            t_sample = t_data

        if scheme_n == 'rand':
            keys = jax.random.split(keyn, num=bs_t+2)
            sample_idx = vmap(get_rand_idx, (0, None, None))(keys, N, bs_n)
            sols_sample = sols_sample[jnp.arange(len(sols_sample))[
                :, None], sample_idx]
        else:
            idx = get_rand_idx(keyn, N, bs_n)
            sols_sample = sols_sample[:, idx]

        if scheme_w == 'dist':
            times = jnp.squeeze(t_sample[1:-1])
            times = jnp.sort(times)
            quad_weights =  0.5 * jnp.concatenate([jnp.array([times[1] - times[0]]),
                                                    (times[2:] - times[:-2]),
                                                    jnp.array([times[-1] - times[-2]])])



        return sols_sample, mu_sample, t_sample, quad_weights

    return args_fn


def interplate_in_t(sols, true_t, interp_t):
    sols = np.asarray(sols)
    T, N, D = sols.shape

    data_spacing = [np.linspace(0.0, 1.0, n) for n in sols.shape[1:]]
    spacing = [np.squeeze(true_t), *data_spacing]

    gt_f = RegularGridInterpolator(
        spacing, sols, method='linear', bounds_error=True)

    interp_spacing = [np.squeeze(interp_t), *data_spacing]
    x_pts = pts_array_from_space(interp_spacing)
    interp_sols = gt_f(x_pts)

    interp_sols = rearrange(interp_sols, '(T N D) -> T N D', N=N, D=D)
    return interp_sols

def dirichlet_indices(key, n_samples, N_index, alpha):
    alphas = jnp.ones(n_samples + 1) * alpha
    spacings = jax.random.dirichlet(key, alphas)
    cumulative = jnp.cumsum(spacings)
    points = cumulative[:-1]
    indices = jnp.asarray(jnp.round(points*N_index,0), dtype=jnp.int32)
    return indices


def arcsin_indices(key, n_samples, N_index):
    alpha, beta = 0.5, 0.5
    points = jax.random.beta(key, alpha, beta, shape=(n_samples,))
    indices = jnp.asarray(jnp.round(points*N_index,0), dtype=jnp.int32)
    return indices



