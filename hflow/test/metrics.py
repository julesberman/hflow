import jax
import jax.numpy as jnp
import numpy as np
import scipy
from einops import rearrange
from jax import jacfwd, jacrev, jit, vmap
from tqdm import tqdm
import matplotlib.pyplot as plt

import hflow.io.result as R
from hflow.config import Test
from hflow.io.utils import log
from hflow.misc.jax import hess_trace_estimator, tracewrap
from scipy.linalg import sqrtm
from hydra.core.hydra_config import HydraConfig
from hflow.config import Config
import jax
import jax.numpy as jnp

from ott import utils
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

def get_cov_diag(A):
    return jnp.diag(jnp.cov(A, rowvar=False))


def compute_metrics(cfg: Config, test_cfg: Test, true_sol, test_sol, mu_i):

    # truncate ic
    true_sol = true_sol[1:]
    test_sol = test_sol[1:]

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
        l2_err_cov = np.linalg.norm(true_cov - test_cov) / np.linalg.norm(true_cov)

        R.RESULT[f"l2_mean_err_{mu_i}"] = l2_mean_err
        R.RESULT[f"l2_cov_err_{mu_i}"] = l2_err_cov

        log.info(f"l2_mean_err {mu_i}: {l2_mean_err:.3e}")
        log.info(f"l2_cov_err {mu_i}: {l2_err_cov:.3e}")

    if test_cfg.electric:

        true_sol_ele = true_sol
        test_sol_ele = test_sol
        boxsize = 50

        if true_sol.shape[-1] > 2:
            true_sol_ele = np.stack([true_sol[:, :, 0], true_sol[:, :, 3]], axis=-1)
            test_sol_ele = np.stack([test_sol[:, :, 0], test_sol[:, :, 3]], axis=-1)
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

        err_electric = np.abs(true_electric - test_electric) / np.abs(true_electric)
        err_electric = np.mean(err_electric)
        R.RESULT[f"err_electric_{mu_i}"] = err_electric
        log.info(f"err_electric_{mu_i}: {err_electric:.3e}")

    if test_cfg.wass:
        log.info(f"computing wasserstein")

        epsilon = test_cfg.w_eps
        n_wass_time = 16

        t_idx = np.linspace(0, 1, n_wass_time, dtype=np.int32)

        w_time = compute_wasserstein_time(test_sol[t_idx], true_sol[t_idx], eps=epsilon)

        R.RESULT[f"time_wass_dist_{mu_i}"] = w_time
        mean_w_dist = np.mean(w_time)
        R.RESULT[f"mean_wass_dist_{mu_i}"] = mean_w_dist
        log.info(f"mean_wass_dist_{mu_i}: {mean_w_dist:.3e}")

    if test_cfg.analytic:
        if cfg.problem == "lin":
            omega = cfg.data.omega
            t_int = np.linspace(0, 1, len(test_sol))
            ms, covs = vmap(analytic_sol, (0, None))(t_int, omega)

            # analytic_w_dist_time = vmap(wass_dist_gauss, (0, 0, 0))(test_sol, ms, covs)
            analytic_w_dist_time = []
            for t, m, c in zip(test_sol, ms, covs):
                analytic_w_dist_time.append(wass_dist_gauss(t, m, c))

            R.RESULT[f"time_analytic_w_dist_{mu_i}"] = analytic_w_dist_time
            mean_aw_dist = np.mean(analytic_w_dist_time)
            R.RESULT[f"mean_analytic_w_dist_{mu_i}"] = mean_aw_dist
            log.info(f"mean_analytic_w_dist_{mu_i}: {mean_aw_dist:.3e}")


def analytic_sol(t, omega):
    sig_0 = 0.1
    mean = jnp.cos(t * omega)
    std = sig_0 * jnp.cos(t * omega)
    return mean, std**2 * jnp.eye(2)


def wass_dist_gauss(sample_points, gaussian_mean, gaussian_cov):
    """
    Calculate the 2-Wasserstein distance between the empirical distribution of sample points
    and a Gaussian distribution with specified mean and covariance.

    Parameters:
    - sample_points: Nx2 numpy array of sampled points.
    - gaussian_mean: 1D numpy array of length 2 representing the mean of the Gaussian.
    - gaussian_cov: 2x2 numpy array representing the covariance matrix of the Gaussian.

    Returns:
    - W2: The 2-Wasserstein distance.
    """

    # Compute the mean and covariance of the sample points
    m1 = jnp.mean(sample_points, axis=0)
    S1 = jnp.cov(sample_points, rowvar=False)

    # Ensure the covariance matrices are symmetric positive definite
    # Add a small regularization term if necessary
    reg = 1e-5 * jnp.eye(2)
    S1 += reg
    gaussian_cov += reg

    # Compute the mean difference term
    mean_diff = m1 - gaussian_mean
    mean_term = jnp.dot(mean_diff, mean_diff)

    # Compute the matrix square root of the sample covariance
    S1_sqrt = sqrtm(S1)

    # Compute the product S1_sqrt * S2 * S1_sqrt
    S_prod = S1_sqrt @ gaussian_cov @ S1_sqrt

    # Compute the square root of the product matrix
    S_prod_sqrt = sqrtm(S_prod)

    # Compute the trace term
    trace_term = jnp.trace(S1 + gaussian_cov - 2 * S_prod_sqrt)

    # Compute the squared 2-Wasserstein distance
    W2_squared = mean_term + trace_term

    # Return the 2-Wasserstein distance
    W2 = jnp.sqrt(W2_squared.real)  # Ensure the result is real

    return W2


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
    from hflow.data.vlasov import get_gradient_matrix, get_laplacian_matrix, getAcc

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