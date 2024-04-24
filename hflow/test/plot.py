import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from jax import jacrev, jit, vmap

from hflow.config import Test
from hflow.io.utils import log
from hflow.misc.plot import imshow_movie, line_movie, scatter_movie


def get_hist(frame, nx=72):
    frame = frame.T
    H, x, y = jnp.histogram2d(
        frame[0], frame[1], bins=nx, range=[[0, 1], [0, 1]])
    return H


get_hist = vmap(jit(get_hist))


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
            plot_sol = jnp.hstack(
                [true_sol[:, idx_sample], test_sol[:, idx_sample]])
            cs = [*['r']*n_plot, *['b']*n_plot]
            scatter_movie(plot_sol, t=t_int, c=cs, alpha=0.3, xlim=[0, 1], ylim=[
                0, 1], show=False, frames=frames, save_to=f'{outdir}/sol_{mu_i}.gif')
        except Exception as e:
            log.error(e, "could not plot particles")

    # plot hist
    if test_cfg.plot_hist:
        try:
            idx_time = np.linspace(0, len(test_sol)-1, frames, dtype=np.int32)
            hist_sol = get_hist(test_sol[idx_time])
            imshow_movie(hist_sol, t=t_int[idx_time], show=False,
                         frames=frames, save_to=f'{outdir}/test_hist_{mu_i}.gif')
            hist_sol = get_hist(true_sol[idx_time])
            imshow_movie(hist_sol, t=t_int[idx_time], show=False,
                         frames=frames, save_to=f'{outdir}/true_hist_{mu_i}.gif')
        except Exception as e:
            log.error(e, "could not plot hist")

    # plot func
    if test_cfg.plot_func:
        true_sol = rearrange(true_sol, 'T N D -> N T D')
        test_sol = rearrange(test_sol, 'T N D -> N T D')
        try:
            ylim = [true_sol.min()*1.1, true_sol.max()*1.1]
            line_movie(true_sol, t=t_int, show=False, ylim=ylim, frames=frames,
                       save_to=f'{outdir}/true_func_{mu_i}.gif')
            line_movie(test_sol, t=t_int, show=False, ylim=ylim, frames=frames,
                       save_to=f'{outdir}/test_func_{mu_i}.gif')
        except Exception as e:
            log.error(e, "could not plot func")
