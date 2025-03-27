import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from jax import jacrev, jit, vmap

from hflow.config import Test
from hflow.io.utils import log
from hflow.misc.plot import  scatter_movie, plot_grid_movie


def get_hist_single(frame, nx):
    frame = frame.T
    H, x, y = jnp.histogram2d(
        frame[0], frame[1], bins=nx)
    return H


def get_hist(frame, nx=100):
    return vmap(get_hist_single, (0, None))(frame, nx)


def plot_test(test_cfg: Test, true_sol, test_sol, t_int, n_plot, mu_i):
    true_sol = np.nan_to_num(true_sol)
    test_sol = np.nan_to_num(test_sol)

    frames = 75
    outdir = HydraConfig.get().runtime.output_dir

    N = test_sol.shape[1]
    n_plot = min(N-1, n_plot)
    idx_sample = np.linspace(0, N-1, n_plot, dtype=np.uint32)

    # plot scatter
    if test_cfg.plot_particles:
        try:
            if true_sol.shape[-1] > 2:
                dim_idx = np.asarray([0, 3])
                true_sol_sub = true_sol[:, :, dim_idx]
                test_sol_sub = test_sol[:, :, dim_idx]
            plot_sol = jnp.hstack(
                [true_sol_sub[:, idx_sample], test_sol_sub[:, idx_sample]])
            cs = [*['r']*n_plot, *['b']*n_plot]
            scatter_movie(plot_sol, t=t_int, c=cs, alpha=0.3, xlim=[0, 1], ylim=[0, 1], 
            show=False, frames=frames, save_to=f'{outdir}/sol_{mu_i}.gif')
        except Exception as e:
            log.error(e, "could not plot particles")

    # plot hist
    if test_cfg.plot_hist:
        try:
            if true_sol.shape[-1] > 2:
                dim_idx = np.asarray([0, 3])
                true_sol_sub = true_sol[:, :, dim_idx]
                test_sol_sub = test_sol[:, :, dim_idx]
            idx_time = np.linspace(0, len(test_sol)-1, frames, dtype=np.int32)
            hist_sol_test = get_hist(test_sol_sub[idx_time])
            hist_sol_true = get_hist(true_sol_sub[idx_time])
            plot_grid_movie([hist_sol_true, hist_sol_test], frames=frames, t=t_int[idx_time], show=False, save_to=f'{outdir}/hist_{mu_i}.gif', titles_x=['True', 'Pred'], live_cbar=False)
        except Exception as e:
            log.error(e, "could not plot hist")

    # plot hist
    if test_cfg.plot_fields:
        try:
            

            xy = true_sol.shape[-1]
            nn = int(xy**0.5)
            true_sol = rearrange(true_sol, 'T N (X Y) -> N T X Y', X=nn)
            test_sol = rearrange(test_sol, 'T N (X Y) -> N T X Y', X=nn)

            plot_grid_movie(true_sol[:9], frames=frames, show=False, save_to=f'{outdir}/true_{mu_i}.gif', live_cbar=True)
            plot_grid_movie(test_sol[:9], frames=frames, show=False, save_to=f'{outdir}/test_{mu_i}.gif', live_cbar=True)

        except Exception as e:
            log.error(e, "could not plot field")
