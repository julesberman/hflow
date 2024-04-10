import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import jacrev, jit, vmap
from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence

import hflow.io.result as R
from hflow.config import Test
from hflow.io.utils import log


def compute_metrics(test_cfg: Test, true_sol, test_sol, mu_i):
    # shape is T N D
    def get_metric(sol):
        mm = np.mean(sol, axis=1)
        var = np.var(sol, axis=1)
        return mm, var

    true_m, true_v = get_metric(true_sol)
    test_m, test_v = get_metric(test_sol)

    t_err_m = np.linalg.norm(true_m - test_m, axis=1) / \
        np.linalg.norm(true_m, axis=1)
    t_err_v = np.linalg.norm(true_v - test_v, axis=1) / \
        np.linalg.norm(true_v, axis=1)

    mean_t_err_m = np.mean(t_err_m)
    mean_t_err_v = np.mean(t_err_v)

    R.RESULT[f'time_mean_err_{mu_i}'] = t_err_m
    R.RESULT[f'time_var_err_{mu_i}'] = t_err_v

    R.RESULT[f'mean_mean_err_{mu_i}'] = mean_t_err_m
    R.RESULT[f'mean_var_err_{mu_i}'] = mean_t_err_v

    log.info(f'mean_mean_err {mu_i}: {mean_t_err_m:.3e}')
    log.info(f'mean_var_err {mu_i}:  {mean_t_err_v:.3e}')


def compute_wasserstein(A, B, eps):
    geometry = pointcloud.PointCloud(x=A, y=B, epsilon=eps)
    result = sinkhorn_divergence.sinkhorn_divergence(geom=geometry, x=A, y=B)
    sk_div = result.divergence
    return jnp.sqrt(sk_div)


def compute_wasserstein_over_D(A, B, eps):
    A_B = jnp.stack([A, B], axis=1)

    def compute(A_B):
        A, B = A_B
        return compute_wasserstein(A, B, eps)

    return vmap(jit(compute))(A_B)