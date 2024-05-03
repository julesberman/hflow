import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import jacrev, jit, vmap
from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence
from tqdm import tqdm

import hflow.io.result as R
from hflow.config import Test
from hflow.io.utils import log


def compute_metrics(test_cfg: Test, true_sol, test_sol, mu_i):

    # truncate ic
    true_sol = true_sol[1:]
    test_sol = test_sol[1:]

    if test_cfg.mean:
        def get_metric(sol):
            mm = np.mean(sol, axis=1)
            var = np.var(sol, axis=1)
            return mm, var

        true_m, true_v = get_metric(true_sol)
        test_m, test_v = get_metric(test_sol)

        t_err_m = np.linalg.norm(true_m - test_m, axis=1) / \
            np.linalg.norm(true_m, axis=1)

        R.RESULT[f'time_mean_true_{mu_i}'] = true_m
        R.RESULT[f'time_mean_test_{mu_i}'] = test_m
        R.RESULT[f'time_mean_err_{mu_i}'] = t_err_m

        mean_t_err_m = np.mean(t_err_m)
        R.RESULT[f'mean_mean_err_{mu_i}'] = mean_t_err_m

        log.info(f'mean_mean_err {mu_i}: {mean_t_err_m:.3e}')

    if test_cfg.electric:
        true_electric = compute_electric_energy(true_sol)
        test_electric = compute_electric_energy(test_sol)
        R.RESULT[f'true_electric_{mu_i}'] = true_electric
        R.RESULT[f'test_electric_{mu_i}'] = test_electric

        err_electric = np.abs(
            true_electric - test_electric) / np.abs(true_electric)
        err_electric = np.mean(err_electric)
        R.RESULT[f'err_electric_{mu_i}'] = err_electric
        log.info(f'err_electric_{mu_i}: {err_electric:.3e}')

    if test_cfg.wass:
        log.info(f'computing wasserstein')
        epsilon = test_cfg.w_eps
        w_time = compute_wasserstein_over_D(true_sol, test_sol, epsilon)

        R.RESULT[f'time_wass_dist_{mu_i}'] = w_time

        mean_wass_dist = np.mean(w_time)
        R.RESULT[f'mean_wass_dist_{mu_i}'] = mean_wass_dist
        log.info(f'mean_wass_dist_{mu_i}: {mean_wass_dist:.3e}')


def compute_wasserstein(A, B, eps):
    geometry = pointcloud.PointCloud(x=A, y=B, epsilon=eps)
    result = sinkhorn_divergence.sinkhorn_divergence(geom=geometry, x=A, y=B)
    sk_div = result.divergence
    return jnp.sqrt(sk_div)


def compute_wasserstein_over_D(A, B, eps):
    T = len(A)
    compute = jit(compute_wasserstein)

    wass = []
    for t in tqdm(range(T)):
        w = compute(A[t], B[t], eps)
        wass.append(w)

    return np.asarray(wass)


def compute_electric_energy(sol):
    from hflow.data.vlasov import (get_gradient_matrix, get_laplacian_matrix,
                                   getAcc)

    T, Nn, D = sol.shape
    N = Nn // 8

    sol = sol[:, :N*8]

    boxsize = 1  # not 50 because everything should be normalized
    sol = np.mod(sol, boxsize)
    sol[sol == 0.0] = 1e-5
    sol[sol == 1.0] = (1-1e-5)

    Lmtx = get_laplacian_matrix(N, boxsize)
    Gmtx = get_gradient_matrix(N, boxsize)

    # this will take essentially as long as the full simulation
    phi = np.zeros((T, Nn))

    for j in range(T):
        _, phi[j, :] = np.squeeze(
            getAcc(sol[j, :, 0][:, None], N, boxsize, 1, Gmtx, Lmtx))

    # calculate electric energy at all times
    E = - 0.5 * np.mean(phi, axis=1)

    true_box = 50

    return E * true_box**2
