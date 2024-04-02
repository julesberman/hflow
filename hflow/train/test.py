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
    true_sol = sol[:, :, samples_idx]
    
    mu_index = 0
  
    true_sol = true_sol[mu_index]
    test_sol = solve_test(s_fn, opt_params, ics[mu_index], t_int,
                        test_cfg.dt, sigma, mus[mu_index], key)
    R.RESULT['true_sol'] = true_sol
    R.RESULT['test_sol'] = test_sol
    R.RESULT['t_int'] = t_int
    
    compute_metrics(true_sol, test_sol)
    
    if test_cfg.plot:
        plot_test(true_sol, test_sol, t_int)
    
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

    # test_sol = solve_sde(drift, diffusion, t_int, get_ic, n_samples, dt=dt, key=key)
    
    return test_sol



def plot_test(true_sol, test_sol, t_int):
    outdir = HydraConfig.get().runtime.output_dir
    n_plot = 500
    plot_sol = jnp.hstack([true_sol, test_sol])
    idx = np.linspace(0, plot_sol.shape[1], n_plot*2, dtype=np.uint32)
    plot_sol = plot_sol[:, idx]
    cs = [*['r']*n_plot,*['b']*n_plot]
    scatter_movie(plot_sol, t=t_int, c=cs, alpha=0.3, xlim=[0,1], ylim=[0,1], show=False, save_to=f'{outdir}/sol.gif')
    
def compute_metrics(true_sol, test_sol):
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
    
    R.RESULT['time_mean_err'] = t_err_m
    R.RESULT['time_var_err'] = t_err_v
    
    R.RESULT['mean_mean_err'] = mean_t_err_m
    R.RESULT['mean_var_err'] = mean_t_err_v
    
    
    log.info(f'mean_mean_err: {mean_t_err_m:.3e}')
    log.info(f'mean_var_err:  {mean_t_err_v:.3e}')