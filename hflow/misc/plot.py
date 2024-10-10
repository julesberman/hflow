from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
# from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable


def imshow_movie(sol, frames=50, t=None, interval=100, tight=False, title='', cmap='viridis', aspect='equal', live_cbar=False, save_to=None, show=True, fps=10):

    fig, ax = plt.subplots()
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    cv0 = sol[0]
    # Here make an AxesImage rather than contour
    im = ax.imshow(cv0, cmap=cmap, aspect=aspect)
    cb = fig.colorbar(im, cax=cax)
    tx = ax.set_title('Frame 0')
    vmax = np.max(sol)
    vmin = np.min(sol)
    ax.set_xticks([])
    ax.set_yticks([])
    if tight:
        plt.tight_layout()

    def animate(frame):
        arr, t = frame
        im.set_data(arr)
        if live_cbar:
            vmax = np.max(arr)
            vmin = np.min(arr)
            im.set_clim(vmin, vmax)
        tx.set_text(f'{title} t={t:.2f}')

    time, w, h = sol.shape
    if t is None:
        t = np.arange(time)
    inc = max(time//frames, 1)
    sol_frames = sol[::inc]
    t_frames = t[::inc]
    frames = list(zip(sol_frames, t_frames))
    ani = FuncAnimation(fig, animate,
                        frames=frames, interval=interval,)
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix('.gif')
        ani.save(p, writer='pillow', fps=fps)

    if show:
        return HTML(ani.to_jshtml())


def imshow_pts_movies(sol, pts, extent, c='r', size=None, alpha=1, frames=50, t=None, interval=100, tight=False, title='', cmap='viridis', aspect='equal', live_cbar=False, save_to=None, show=True, fps=30):

    fig, ax = plt.subplots()
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    cv0 = sol[0]
    # Here make an AxesImage rather than contour
    im = ax.imshow(cv0, cmap=cmap, aspect=aspect, extent=extent)
    sct = ax.scatter(x=pts[0, 0], y=pts[0, 1], alpha=alpha, s=size, c=c)
    cb = fig.colorbar(im, cax=cax)
    tx = ax.set_title('Frame 0')
    vmax = np.max(sol)
    vmin = np.min(sol)

    if tight:
        plt.tight_layout()

    def animate(frame):
        (arr, scatter), t = frame
        im.set_data(arr)
        im.set_extent(extent)
        sct.set_offsets(scatter.T)
        if live_cbar:
            vmax = np.max(arr)
            vmin = np.min(arr)
            im.set_clim(vmin, vmax)
        tx.set_text(f'{title} t={t:.2f}')

    time, w, h = sol.shape
    if t is None:
        t = np.arange(time)
    inc = max(time//frames, 1)
    sol_frames = sol[::inc]
    t_frames = t[::inc]
    pts = pts[::inc]
    sol_frames = zip(sol_frames, pts)
    frames = list(zip(sol_frames, t_frames))
    ani = FuncAnimation(fig, animate,
                        frames=frames, interval=interval,)
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix('.gif')
        ani.save(p, writer='pillow', fps=fps)

    if show:
        return HTML(ani.to_jshtml())


def line_movie(sol, frames=50, t=None, x=None,  title='', interval=100, ylim=None, save_to=None, show=True, legend=None, tight=False):
    sol = np.asarray(sol)
    if len(sol.shape) == 2:
        sol = np.expand_dims(sol, axis=0)

    n_lines, time, space = sol.shape
    sol = rearrange(sol, 'l t s -> t s l')
    fig, ax = plt.subplots()
    ax.set_ylim([sol.min(), sol.max()])
    if ylim is not None:
        ax.set_ylim(ylim)
    if x is None:
        x = np.arange(sol.shape[1])

    cycler = plt.cycler(
        linestyle=['-', '--']*5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'])
    ax.set_prop_cycle(cycler)
    line = ax.plot(x, sol[0], )
    if tight:
        plt.tight_layout()

    if legend is not None:
        ax.legend(legend)

    def animate(frame):
        sol, t = frame
        ax.set_title(f'{title} t={t:.3f}')
        for i, l in enumerate(line):
            l.set_ydata(sol[:, i])
        return line

    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return line,

    if t is None:
        t = np.arange(time)
    inc = max(time//frames, 1)
    sol_frames = sol[::inc]
    t_frames = t[::inc]
    sol_frames = sol[::inc]
    frames = list(zip(sol_frames, t_frames))
    ani = FuncAnimation(fig, animate, frames=frames,
                        interval=interval, blit=True)
    plt.close()
    if save_to is not None:
        p = Path(save_to).with_suffix('.gif')
        ani.save(p, writer='pillow', fps=30)

    if show:
        return HTML(ani.to_jshtml())


def scatter_movie(pts, c='r', size=None, xlim=None, ylim=None, alpha=1, frames=60, t=None, title='', interval=100, save_to=None, show=True, fps=10):
    pts = rearrange(pts, 't n d -> t d n')
    fig, ax = plt.subplots()

    sct = ax.scatter(x=pts[0, 0], y=pts[0, 1], alpha=alpha, s=size, c=c)
    mm = pts.min(axis=(0, 2))
    mx = pts.max(axis=(0, 2))

    if xlim is None:
        xlim = [mm[0], mx[0]]
    if ylim is None:
        ylim = [mm[1], mx[1]]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    tx = ax.set_title('Frame 0')

    def animate(frame):
        scatter, t = frame
        sct.set_offsets(scatter.T)
        tx.set_text(f'{title} t={t:.2f}')

    time = len(pts)
    if t is None:
        t = np.arange(time)
    inc = max(time//frames, 1)
    t_frames = t[::inc]
    pts = pts[::inc]

    frames = list(zip(pts, t_frames))
    ani = FuncAnimation(fig, animate,
                        frames=frames, interval=interval,)
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix('.gif')
        ani.save(p, writer='pillow', fps=fps)

    if show:
        return HTML(ani.to_jshtml())


def line_movie(sol, frames=50, t=None, x=None, color=None, title='', interval=100, ylim=None, save_to=None, show=True, legend=None, tight=False, fps=10):
    sol = np.asarray(sol)
    if len(sol.shape) == 2:
        sol = np.expand_dims(sol, axis=0)

    n_lines, time, space = sol.shape
    sol = rearrange(sol, 'l t s -> t s l')
    fig, ax = plt.subplots()
    ax.set_ylim([sol.min(), sol.max()])
    if ylim is not None:
        ax.set_ylim(ylim)
    if x is None:
        x = np.arange(sol.shape[1])

    if color is not None:
        cycler = plt.cycler(color=color)
        ax.set_prop_cycle(cycler)
    line = ax.plot(x, sol[0], )
    if tight:
        plt.tight_layout()

    if legend is not None:
        ax.legend(legend)

    def animate(frame):
        sol, t = frame
        ax.set_title(f'{title} t={t:.3f}')
        for i, l in enumerate(line):
            l.set_ydata(sol[:, i])
        return line

    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return line,

    if t is None:
        t = np.arange(time)
    inc = max(time//frames, 1)
    sol_frames = sol[::inc]
    t_frames = t[::inc]
    sol_frames = sol[::inc]
    frames = list(zip(sol_frames, t_frames))
    ani = FuncAnimation(fig, animate, frames=frames,
                        interval=interval, blit=True)
    plt.close()
    if save_to is not None:
        p = Path(save_to).with_suffix('.gif')
        ani.save(p, writer='pillow', fps=fps)

    if show:
        return HTML(ani.to_jshtml())


def trajectory_movie(y, frames=50, title='', ylabel='', xlabel='Time', legend=[], x=None, interval=100, ylim=None, save_to=None):

    y = np.asarray(y)
    if x is None:
        x = np.arange(len(y))

    fig, ax = plt.subplots()
    total = len(x)
    inc = max(total//frames, 1)
    x = x[::inc]
    y = y[::inc]
    if ylim is None:
        ylim = np.array([y.min(), y.max()])
    xlim = [x.min(), x.max()]

    def animate(i):
        ax.cla()
        ax.plot(x[:i], y[:i], marker='o', markevery=[-1])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(legend, loc='lower right')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} t={x[i]:.2f}')

    ani = FuncAnimation(fig, animate, frames=len(x), interval=interval)
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix('.gif')
        ani.save(p, writer='pillow', fps=30)

    return HTML(ani.to_jshtml())
