import hflow.io.result as R
from hflow.config import Config
from jax import vmap, jit, jacrev
import jax.numpy as jnp

from einops import rearrange
import jax
import matplotlib.pyplot as plt
from hflow.data.sde import solve_sde_ic
from hflow.misc.jax import get_rand_idx
from hflow.misc.plot import scatter_movie
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hflow.io.utils import log

def test_model(cfg: Config, data, s_fn, opt_params, key):
    test_cfg = cfg.test
    sol, mus, t = data
    t_int = np.linspace(0.0, 1.0, len(t))
    M, T, N, D = sol.shape
    samples_idx = get_rand_idx(key, sol.shape[2], test_cfg.n_samples)
    ics = sol[:, 0, samples_idx, :]

    sigma = cfg.loss.sigma
    sol = sol[:, :, samples_idx]
    
    R.RESULT['t_int'] = t_int
    for mu_i in range(len(mus)):
        true_sol = sol[mu_i]
        test_sol = solve_test(s_fn, opt_params, ics[mu_i], t_int,
                            test_cfg.dt, sigma, mus[mu_i], key)
        R.RESULT[f'true_sol_{mu_i}'] = true_sol
        R.RESULT[f'test_sol_{mu_i}'] = test_sol
     
        compute_metrics(true_sol, test_sol, mu_i)
        
        if test_cfg.plot:
            plot_test(true_sol, test_sol, t_int, test_cfg.plot_samples, mu_i)
        
    return test_sol


def solve_test(s_fn, params, ics, t_int, dt, sigma, mu, key):
    s_dx = jacrev(s_fn, 1)

    
    def drift(t, y, *args):
        mu_t = jnp.concatenate([mu, t.reshape(1)])
        f = jnp.squeeze(s_dx(mu_t, y, params))
        return f

    def diffusion(t, y, *args):
        return sigma * jnp.ones_like(y)
    
    keys = jax.random.split(key, num=len(ics))
    test_sol = vmap(solve_sde_ic, (0, 0, None, None, None, None))(ics, keys, t_int, dt, drift, diffusion)
    test_sol = rearrange(test_sol, 'N T D -> T N D')

    return test_sol



def plot_test(true_sol, test_sol, t_int, n_plot, mu_i):
    outdir = HydraConfig.get().runtime.output_dir
    N = test_sol.shape[1]
    n_plot = min(N-1, n_plot)
    idx = np.linspace(0, N-1, n_plot, dtype=np.uint32)
    plot_sol = jnp.hstack([true_sol[:, idx], test_sol[:, idx]])
  
    cs = [*['r']*n_plot,*['b']*n_plot]
    scatter_movie(plot_sol, t=t_int, c=cs, alpha=0.3, xlim=[0,1], ylim=[0,1], show=False, save_to=f'{outdir}/sol_{mu_i}.gif')
    
def compute_metrics(true_sol, test_sol, mu_i):
    # shape is T N D
    def get_metric(sol):
        mm = np.mean(sol, axis=1)
        var = np.var(sol, axis=1)
        return mm, var
    
    true_m, true_v = get_metric(true_sol)
    test_m, test_v = get_metric(test_sol)
    
    t_err_m = np.linalg.norm(true_m - test_m, axis=1) / np.linalg.norm(true_m,axis=1)
    t_err_v = np.linalg.norm(true_v - test_v, axis=1) / np.linalg.norm(true_v,axis=1)
    
    mean_t_err_m = np.mean(t_err_m)
    mean_t_err_v = np.mean(t_err_v)
    
    R.RESULT[f'time_mean_err_{mu_i}'] = t_err_m
    R.RESULT[f'time_var_err_{mu_i}'] = t_err_v
    
    R.RESULT[f'mean_mean_err_{mu_i}'] = mean_t_err_m
    R.RESULT[f'mean_var_err_{mu_i}'] = mean_t_err_v
    
    
    log.info(f'mean_mean_err {mu_i}: {mean_t_err_m:.3e}')
    log.info(f'mean_var_err {mu_i}:  {mean_t_err_v:.3e}')