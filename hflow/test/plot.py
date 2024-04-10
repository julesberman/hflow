import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from jax import jacrev, jit, vmap

from hflow.config import Test
from hflow.misc.plot import imshow_movie, scatter_movie


def get_hist(frame, nx=72):
    frame = frame.T
    H, x, y = jnp.histogram2d(
        frame[0], frame[1], bins=nx, range=[[0, 1], [0, 1]])
    return H


get_hist = vmap(jit(get_hist))


def plot_test(test_cfg: Test, true_sol, test_sol, t_int, n_plot, mu_i):
    frames = 75
    outdir = HydraConfig.get().runtime.output_dir

    # plot scatter
    N = test_sol.shape[1]
    n_plot = min(N-1, n_plot)
    idx = np.linspace(0, N-1, n_plot, dtype=np.uint32)
    plot_sol = jnp.hstack([true_sol[:, idx], test_sol[:, idx]])
    cs = [*['r']*n_plot, *['b']*n_plot]
    scatter_movie(plot_sol, t=t_int, c=cs, alpha=0.3, xlim=[0, 1], ylim=[
                  0, 1], show=False, frames=frames, save_to=f'{outdir}/sol_{mu_i}.gif')

    # plot hist
    idx = np.linspace(0, len(test_sol)-1, frames, dtype=np.int32)
    hist_sol = get_hist(test_sol[idx])
    imshow_movie(hist_sol, t=t_int[idx], show=False,
                 frames=frames, save_to=f'{outdir}/test_hist_{mu_i}.gif')
