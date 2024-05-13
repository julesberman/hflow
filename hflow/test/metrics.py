import jax
import jax.numpy as jnp
import numpy as np
import scipy
from einops import rearrange
from jax import jacfwd, jacrev, jit, vmap
from tqdm import tqdm

import hflow.io.result as R
from hflow.config import Test
from hflow.io.utils import log
from hflow.misc.jax import batchmap, hess_trace_estimator, tracewrap


def get_cov_diag(A):
    return jnp.diag(jnp.cov(A, rowvar=False))


def compute_metrics(test_cfg: Test, true_sol, test_sol, mu_i):

    # truncate ic
    true_sol = true_sol[1:]
    test_sol = test_sol[1:]

    (d_shift, d_scale) = R.RESULT['data_norm']
    d_shift, d_scale = d_shift[0], d_scale[0]

    true_sol = (true_sol*d_scale)+d_shift
    test_sol = (test_sol*d_scale)+d_shift

    if test_cfg.mean:
        def get_metric(sol):
            mm = np.mean(sol, axis=(1,))
            cov = vmap(get_cov_diag)(sol)
            return mm, cov

        true_m, true_cov = get_metric(true_sol)
        test_m, test_cov = get_metric(test_sol)

        R.RESULT[f'full_mean_true_{mu_i}'] = true_m
        R.RESULT[f'full_cov_true_{mu_i}'] = true_cov
        R.RESULT[f'full_mean_test_{mu_i}'] = test_m
        R.RESULT[f'full_cov_test_{mu_i}'] = test_cov

        time_err_m = np.linalg.norm(
            true_m - test_m, axis=(-1)) / np.linalg.norm(true_m, axis=(-1))
        time_err_cov = np.linalg.norm(
            true_cov - test_cov, axis=(-1)) / np.linalg.norm(true_cov, axis=(-1))

        R.RESULT[f'time_mean_err_{mu_i}'] = time_err_m
        R.RESULT[f'time_cov_err_{mu_i}'] = time_err_cov

        mean_err = np.mean(time_err_m)
        cov_err = np.mean(time_err_cov)

        R.RESULT[f'mean_mean_err_{mu_i}'] = mean_err
        R.RESULT[f'mean_cov_err_{mu_i}'] = cov_err

        l2_mean_err = np.linalg.norm(true_m - test_m) / np.linalg.norm(true_m)
        l2_err_cov = np.linalg.norm(
            true_cov - test_cov) / np.linalg.norm(true_cov)

        R.RESULT[f'l2_mean_err_{mu_i}'] = l2_mean_err
        R.RESULT[f'l2_cov_err_{mu_i}'] = l2_err_cov

        log.info(f'l2_mean_err {mu_i}: {l2_mean_err:.3e}')
        log.info(f'l2_cov_err {mu_i}: {l2_err_cov:.3e}')

    if test_cfg.electric:

        true_sol_ele = true_sol
        test_sol_ele = test_sol
        boxsize = 50

        if true_sol.shape[-1] > 2:
            true_sol_ele = np.stack(
                [true_sol[:, :, 0], true_sol[:, :, 3]], axis=-1)
            test_sol_ele = np.stack(
                [test_sol[:, :, 0], test_sol[:, :, 3]], axis=-1)
            boxsize = 4*np.pi

        true_electric = compute_electric_energy(true_sol_ele, boxsize=boxsize)
        test_electric = compute_electric_energy(test_sol_ele, boxsize=boxsize)
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
        s_time = compute_wasserstein_over_D(true_sol, test_sol, epsilon)

        R.RESULT[f'time_sink_dist_{mu_i}'] = s_time
        mean_sink_dist = np.mean(s_time)
        R.RESULT[f'mean_sink_dist_{mu_i}'] = mean_sink_dist
        log.info(f'mean_sink_dist_{mu_i}: {mean_sink_dist:.3e}')


def compute_wasserstein_scipy(A, B):

    T = len(A)

    wass = []
    for t in tqdm(range(T)):
        at, bt = A[t], B[t]
        at, bt = at[:1000], bt[:1000]
        w = scipy.stats.wasserstein_distance_nd(at, bt)
        wass.append(w)

    return wass


def compute_wasserstein_ott(A, B, eps):
    from ott.geometry import pointcloud
    from ott.tools import sinkhorn_divergence
    geometry = pointcloud.PointCloud(x=A, y=B, epsilon=eps)
    result = sinkhorn_divergence.sinkhorn_divergence(geom=geometry, x=A, y=B)
    sk_div = result.divergence
    return sk_div


def compute_wasserstein_over_D(A, B, eps):
    T = len(A)
    compute = jit(compute_wasserstein_ott)

    wass = []
    for t in tqdm(range(T)):
        w = compute(A[t], B[t], eps)
        wass.append(w)

    return np.asarray(wass)


def compute_electric_energy(sol, boxsize=50):
    from hflow.data.vlasov import (get_gradient_matrix, get_laplacian_matrix,
                                   getAcc)

    T, Nn, D = sol.shape
    N = Nn // 8

    sol = sol[:, :N*8]

    sol = np.mod(sol, boxsize)
    # sol[sol == 0.0] = 1e-5
    # sol[sol == 1.0] = (1-1e-5)

    Lmtx = get_laplacian_matrix(N, boxsize)
    Gmtx = get_gradient_matrix(N, boxsize)

    # this will take essentially as long as the full simulation
    phi = np.zeros((T, Nn))

    for j in range(T):
        _, phi[j, :] = np.squeeze(
            getAcc(sol[j, :, 0][:, None], N, boxsize, 1, Gmtx, Lmtx))

    # calculate electric energy at all times
    E = - 0.5 * np.mean(phi, axis=1)

    return E


# def log_rho_0(x): return -jnp.sum(x**2)/2 / \
#     SD_0**2 - D/2*jnp.log(2*jnp.pi*SD_0**2)


# def get_entropy(sols, t_eval, s_fn, log_rho_0):
#     # s is a funcion t, x -> R
#     # log_rho_0 is a function x -> R
#     # laplace_s is a function t, x -> R
#     # t_eval is the array of times where the solution is evaluated
#     s_dx = jacrev(s, 1)
#     trace_dds = tracewrap(jacfwd(s_dx, 1))
#     laplace_s = vmap(trace_dds, (None, 0, None))

#     T, N, D = sols.shape
#     E_log_rho_0 = jnp.sum(vmap(log_rho_0)(sols[0, :, :])) / N
#     sols_with_t = jnp.concatenate(
#         [t_eval[:, None, None] * jnp.ones((T, N, 1)), sols], axis=2)
#     weights = - t_eval[:-1] + t_eval[1:]  # get the dts
#     sols_with_t = sols_with_t[:-1, :, :]  # cut off the last time
#     sols_with_t_and_weights = jnp.concatenate(
#         [weights[:, None, None] * jnp.ones((T-1, N, 1)), sols_with_t], axis=2)

#     def integrand(wtx):
#         w, t, x = wtx[0], wtx[1], wtx[2:]
#         return w * laplace_s(t, x)

#     integrand_all_times = jnp.sum(
#         vmap(vmap(integrand))(sols_with_t_and_weights)) / N

#     return - E_log_rho_0 + integrand_all_times
