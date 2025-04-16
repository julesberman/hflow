import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from jax import jacfwd, jacrev, jit, vmap
from ott import utils
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from scipy.linalg import sqrtm
from tqdm import tqdm

import hflow.io.result as R
from hflow.config import Config, Test
from hflow.io.utils import log
from hflow.misc.jax import hess_trace_estimator, tracewrap


def get_cov_diag(A):
    return jnp.diag(jnp.cov(A, rowvar=False))


def compute_metrics(cfg: Config, test_cfg: Test, true_sol, test_sol, t_eval, s_fn, opt_params, mu, mu_i):

    # truncate ic
    true_sol = true_sol[1:]
    test_sol = test_sol[1:]
    t_eval = t_eval[1:]

    true_og = true_sol

    (d_shift, d_scale) = R.RESULT["data_norm"]
    if cfg.data.normalize:
        d_shift, d_scale = d_shift[0], d_scale[0]
        true_sol = (true_sol * d_scale) + d_shift
        test_sol = (test_sol * d_scale) + d_shift

    if test_cfg.mean:

        def get_metric(sol):
            mm = np.mean(sol, axis=(1,))
            cov = vmap(get_cov_diag)(sol)
            return mm, cov

        true_m, true_cov = get_metric(true_sol)
        test_m, test_cov = get_metric(test_sol)

        R.RESULT[f"full_mean_true_{mu_i}"] = true_m
        R.RESULT[f"full_cov_true_{mu_i}"] = true_cov
        R.RESULT[f"full_mean_test_{mu_i}"] = test_m
        R.RESULT[f"full_cov_test_{mu_i}"] = test_cov

        time_err_m = np.linalg.norm(true_m - test_m, axis=(-1)) / np.linalg.norm(
            true_m, axis=(-1)
        )
        time_err_cov = np.linalg.norm(true_cov - test_cov, axis=(-1)) / np.linalg.norm(
            true_cov, axis=(-1)
        )

        R.RESULT[f"time_mean_err_{mu_i}"] = time_err_m
        R.RESULT[f"time_cov_err_{mu_i}"] = time_err_cov

        mean_err = np.mean(time_err_m)
        cov_err = np.mean(time_err_cov)

        R.RESULT[f"mean_mean_err_{mu_i}"] = mean_err
        R.RESULT[f"mean_cov_err_{mu_i}"] = cov_err

        l2_mean_err = np.linalg.norm(true_m - test_m) / np.linalg.norm(true_m)
        l2_err_cov = np.linalg.norm(
            true_cov - test_cov) / np.linalg.norm(true_cov)

        R.RESULT[f"l2_mean_err_{mu_i}"] = l2_mean_err
        R.RESULT[f"l2_cov_err_{mu_i}"] = l2_err_cov

        log.info(f"l2_mean_err {mu_i}: {l2_mean_err:.3e}")
        log.info(f"l2_cov_err {mu_i}: {l2_err_cov:.3e}")

        if 'all_l2_mean_err' not in R.RESULT:
            R.RESULT['all_l2_mean_err'] = [l2_mean_err]
        else:
            R.RESULT['all_l2_mean_err'].append(l2_mean_err)

        if 'all_l2_cov_err' not in R.RESULT:
            R.RESULT['all_l2_cov_err'] = [l2_err_cov]
        else:
            R.RESULT['all_l2_cov_err'].append(l2_err_cov)

    if test_cfg.electric:

        true_sol_ele = true_sol
        test_sol_ele = test_sol
        boxsize = 50

        if true_sol.shape[-1] > 2:
            true_sol_ele = np.stack(
                [true_sol[:, :, 0], true_sol[:, :, 3]], axis=-1)
            test_sol_ele = np.stack(
                [test_sol[:, :, 0], test_sol[:, :, 3]], axis=-1)
            boxsize = 4 * np.pi

        true_electric = compute_electric_energy(true_sol_ele, boxsize=boxsize)
        test_electric = compute_electric_energy(test_sol_ele, boxsize=boxsize)
        R.RESULT[f"true_electric_{mu_i}"] = true_electric
        R.RESULT[f"test_electric_{mu_i}"] = test_electric

        outdir = HydraConfig.get().runtime.output_dir
        t_int = np.linspace(0, 1, len(test_electric))
        plt.semilogy(t_int, true_electric, label="True")
        plt.semilogy(t_int, test_electric, label="Test")
        plt.xlabel("time")
        plt.ylabel("electric energy")
        plt.savefig(f"{outdir}/electric_{mu_i}.png")
        plt.clf()

        err_electric = np.abs(
            true_electric - test_electric) / np.abs(true_electric)
        err_electric = np.mean(err_electric)
        R.RESULT[f"err_electric_{mu_i}"] = err_electric
        log.info(f"err_electric_{mu_i}: {err_electric:.3e}")

        if 'all_err_electric' not in R.RESULT:
            R.RESULT['all_err_electric'] = [err_electric]
        else:
            R.RESULT['all_err_electric'].append(err_electric)

    if test_cfg.wass:
        log.info(f"computing wasserstein")

        epsilon = test_cfg.w_eps
        n_wass_time = 16

        t_idx = np.linspace(0, 1, n_wass_time, dtype=np.int32)

        n_sample_wass = 5_000

        test_sol_wass = test_sol[t_idx, :n_sample_wass]
        true_sol_wass = true_sol[t_idx, :n_sample_wass]

        w_time = compute_wasserstein_time(
            test_sol_wass, true_sol_wass, eps=epsilon)

        R.RESULT[f"time_wass_dist_{mu_i}"] = w_time
        mean_w_dist = np.mean(w_time)
        R.RESULT[f"mean_wass_dist_{mu_i}"] = mean_w_dist
        log.info(f"mean_wass_dist_{mu_i}: {mean_w_dist:.3e}")

        if 'all_wass_dist' not in R.RESULT:
            R.RESULT['all_wass_dist'] = [mean_w_dist]
        else:
            R.RESULT['all_wass_dist'].append(mean_w_dist)

    if test_cfg.analytic:
        from hflow.data.pot import combined_potential

        def grad_V(x, t):
            return -jax.grad(combined_potential)(x, t)
        grad_V_Vx = vmap(grad_V, (0, None))
        grad_V_Vx_Vt = vmap(grad_V_Vx, (0, 0))

        true_V = grad_V_Vx_Vt(true_sol, t_eval)
        # true_V = rearrange(true_V, 'T D N -> N T D')

        s_dx = jacrev(s_fn, 1)
        mu_v = jnp.asarray([mu]).reshape(1)

        def grad_S(x, t):
            t = t.reshape(1)
            mu_t = jnp.concatenate([mu_v, t])
            f = jnp.squeeze(s_dx(mu_t, x, opt_params))
            return f

        grad_S_Vx = vmap(grad_S, (0, None))
        grad_S_Vx_Vt = vmap(grad_S_Vx, (0, 0))
        true_S = grad_S_Vx_Vt(true_og, t_eval)

        err_n_time = jnp.linalg.norm(
            true_V - true_S, axis=-1)/jnp.linalg.norm(true_V, axis=-1)
        err_time = err_n_time.mean(axis=-1)
        mean_err = err_time.mean()

        R.RESULT[f"time_analytic_{mu_i}"] = err_time
        R.RESULT[f"err_analytic_{mu_i}"] = mean_err
        log.info(f"err_analytic_{mu_i}: {mean_err:.3e}")


def average_metrics(mus):

    # compute averages
    metrics = [
        "err_electric",
        "test_integrate_time",
        "FOM_integrate_time",
        "mean_sink_dist",
    ]
    for metric in metrics:
        count = 0
        total = 0
        for mu_i in range(len(mus)):
            key = f"metric_{mu_i}"
            if key in R.RESULT:
                total += R.RESULT[key]
                count += 1

        if count > 0:
            R.RESULT[f"{metric}_total"] = total / count


def compute_electric_energy(sol, boxsize=50):
    from hflow.data.vlasov import (get_gradient_matrix, get_laplacian_matrix,
                                   getAcc)

    T, Nn, D = sol.shape
    N = Nn // 8

    sol = sol[:, : N * 8]

    sol = np.mod(sol, boxsize)
    # sol[sol == 0.0] = 1e-5
    # sol[sol == 1.0] = (1-1e-5)

    Lmtx = get_laplacian_matrix(N, boxsize)
    Gmtx = get_gradient_matrix(N, boxsize)

    # this will take essentially as long as the full simulation
    phi = np.zeros((T, Nn))

    for j in range(T):
        _, phi[j, :] = np.squeeze(
            getAcc(sol[j, :, 0][:, None], N, boxsize, 1, Gmtx, Lmtx)
        )

    # calculate electric energy at all times
    E = -0.5 * np.mean(phi, axis=1)

    return E


def compute_wasserstein_time(test_sol, true_sol, eps=1e-3):
    T = test_sol.shape[0]
    wdist = []
    solver = sinkhorn.Sinkhorn(max_iterations=10_000, threshold=eps)
    solver = jit(solver)

    for t in tqdm(range(T), colour='blue'):
        x = test_sol[t]
        y = true_sol[t]
        geom = pointcloud.PointCloud(x, y)
        lp = linear_problem.LinearProblem(geom)
        out = solver(lp)

        wdist_t = jnp.sqrt(out.primal_cost)
        wdist.append(wdist_t)

    # Stack into a single JAX array
    return jnp.stack(wdist, axis=0)
