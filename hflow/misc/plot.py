from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from IPython.display import HTML
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable


def imshow_movie(
    sol,
    frames=50,
    t=None,
    interval=100,
    tight=False,
    title="",
    cmap="viridis",
    aspect="equal",
    live_cbar=True,
    save_to=None,
    show=True,
    fps=10,
    c_norm=None,
    t_txt=True
):

    fig, ax = plt.subplots()
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")

    cv0 = sol[0]

    if c_norm is not None:
        norm = colors.Normalize(vmin=c_norm[0], vmax=c_norm[1])
    else:
        norm = colors.Normalize(vmin=np.min(sol), vmax=np.max(sol))

    # Here make an AxesImage rather than contour
    im = ax.imshow(cv0, cmap=cmap, aspect=aspect, norm=norm)
    cb = fig.colorbar(im, cax=cax)
    if t_txt:
        tx = ax.set_title("Frame 0")
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
        if t_txt:
            tx.set_text(f"{title} t={t:.2f}")

    time = sol.shape[0]
    if t is None:
        t = np.arange(time)
    inc = max(time // frames, 1)
    sol_frames = sol[::inc]
    t_frames = t[::inc]
    frames = list(zip(sol_frames, t_frames))
    ani = FuncAnimation(
        fig,
        animate,
        frames=frames,
        interval=interval,
    )
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix(".gif")
        ani.save(p, writer="pillow", fps=fps)

    if show:
        return HTML(ani.to_jshtml())


def imshow_pts_movies(
    sol,
    pts,
    extent,
    c="r",
    size=None,
    alpha=1,
    frames=50,
    t=None,
    interval=100,
    tight=False,
    title="",
    cmap="viridis",
    aspect="equal",
    live_cbar=False,
    save_to=None,
    show=True,
    fps=30,
):

    fig, ax = plt.subplots()
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")

    cv0 = sol[0]
    # Here make an AxesImage rather than contour
    im = ax.imshow(cv0, cmap=cmap, aspect=aspect, extent=extent)
    sct = ax.scatter(x=pts[0, 0], y=pts[0, 1], alpha=alpha, s=size, c=c)
    cb = fig.colorbar(im, cax=cax)
    tx = ax.set_title("Frame 0")
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
        tx.set_text(f"{title} t={t:.2f}")

    time, w, h = sol.shape
    if t is None:
        t = np.arange(time)
    inc = max(time // frames, 1)
    sol_frames = sol[::inc]
    t_frames = t[::inc]
    pts = pts[::inc]
    sol_frames = zip(sol_frames, pts)
    frames = list(zip(sol_frames, t_frames))
    ani = FuncAnimation(
        fig,
        animate,
        frames=frames,
        interval=interval,
    )
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix(".gif")
        ani.save(p, writer="pillow", fps=fps)

    if show:
        return HTML(ani.to_jshtml())


def line_movie(
    sol,
    frames=50,
    t=None,
    x=None,
    title="",
    interval=100,
    ylim=None,
    save_to=None,
    show=True,
    legend=None,
    tight=False,
):
    sol = np.asarray(sol)
    if len(sol.shape) == 2:
        sol = np.expand_dims(sol, axis=0)

    n_lines, time, space = sol.shape
    sol = rearrange(sol, "l t s -> t s l")
    fig, ax = plt.subplots()
    ax.set_ylim([sol.min(), sol.max()])
    if ylim is not None:
        ax.set_ylim(ylim)
    if x is None:
        x = np.arange(sol.shape[1])

    cycler = plt.cycler(
        linestyle=["-", "--"] * 5,
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"],
    )
    ax.set_prop_cycle(cycler)
    line = ax.plot(
        x,
        sol[0],
    )
    if tight:
        plt.tight_layout()

    if legend is not None:
        ax.legend(legend)

    def animate(frame):
        sol, t = frame
        ax.set_title(f"{title} t={t:.3f}")
        for i, l in enumerate(line):
            l.set_ydata(sol[:, i])
        return line

    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return (line,)

    if t is None:
        t = np.arange(time)
    inc = max(time // frames, 1)
    sol_frames = sol[::inc]
    t_frames = t[::inc]
    sol_frames = sol[::inc]
    frames = list(zip(sol_frames, t_frames))
    ani = FuncAnimation(fig, animate, frames=frames,
                        interval=interval, blit=True)
    plt.close()
    if save_to is not None:
        p = Path(save_to).with_suffix(".gif")
        ani.save(p, writer="pillow", fps=30)

    if show:
        return HTML(ani.to_jshtml())


def scatter_movie(
    pts,
    c="r",
    n_samples=None,
    size=None,
    xlim=None,
    ylim=None,
    alpha=1,
    frames=60,
    t=None,
    title="",
    interval=100,
    save_to=None,
    show=True,
    fps=10,
    ticks=True
):
    pts = np.asarray(pts)

    if len(pts.shape) == 4:
        g, _, n, _ = pts.shape
        c = []
        colors = ["r", "b", "g", "m", "k"]
        for i in range(g):
            c.extend([colors[i]] * n)
        pts = rearrange(pts, "g t n d -> t (g n) d")

    pts = rearrange(pts, "t n d -> t d n")

    if n_samples is not None:
        sample_idx = np.random.choice(
            pts.shape[-1] - 1, size=n_samples, replace=False)
        sample_idx = np.asarray(sample_idx, dtype=np.int32)
        pts = pts[:, :, sample_idx]
        if type(c) == list:
            c = c[sample_idx]
    fig, ax = plt.subplots()

    sct = ax.scatter(x=pts[0, 0], y=pts[0, 1],
                     alpha=alpha, s=size, c=c)
    mm = pts.min(axis=(0, 2))
    mx = pts.max(axis=(0, 2))

    if xlim is None:
        xlim = [mm[0], mx[0]]
    if ylim is None:
        ylim = [mm[1], mx[1]]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    if title is not None:
        tx = ax.set_title("Frame 0")

    if not ticks:
        plt.xticks([])
        plt.yticks([])

    def animate(frame):
        scatter, t = frame
        sct.set_offsets(scatter.T)
        if title is not None:
            tx.set_text(f"{title} t={t:.2f}")

    time = len(pts)
    if t is None:
        t = np.arange(time)
    inc = max(time // frames, 1)
    t_frames = t[::inc]
    pts = pts[::inc]

    frames = list(zip(pts, t_frames))
    ani = FuncAnimation(
        fig,
        animate,
        frames=frames,
        interval=interval,
    )
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix(".gif")
        ani.save(p, writer="pillow", fps=fps)

    if show:
        return HTML(ani.to_jshtml())


def line_movie(
    sol,
    frames=50,
    t=None,
    x=None,
    color=None,
    title="",
    interval=100,
    ylim=None,
    save_to=None,
    show=True,
    legend=None,
    tight=False,
    fps=10,
):
    sol = np.asarray(sol)
    if len(sol.shape) == 2:
        sol = np.expand_dims(sol, axis=0)

    n_lines, time, space = sol.shape
    sol = rearrange(sol, "l t s -> t s l")
    fig, ax = plt.subplots()
    ax.set_ylim([sol.min(), sol.max()])
    if ylim is not None:
        ax.set_ylim(ylim)
    if x is None:
        x = np.arange(sol.shape[1])

    if color is not None:
        cycler = plt.cycler(color=color)
        ax.set_prop_cycle(cycler)
    line = ax.plot(
        x,
        sol[0],
    )
    if tight:
        plt.tight_layout()

    if legend is not None:
        ax.legend(legend)

    def animate(frame):
        sol, t = frame
        ax.set_title(f"{title} t={t:.3f}")
        for i, l in enumerate(line):
            l.set_ydata(sol[:, i])
        return line

    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return (line,)

    if t is None:
        t = np.arange(time)
    inc = max(time // frames, 1)
    sol_frames = sol[::inc]
    t_frames = t[::inc]
    sol_frames = sol[::inc]
    frames = list(zip(sol_frames, t_frames))
    ani = FuncAnimation(fig, animate, frames=frames,
                        interval=interval, blit=True)
    plt.close()
    if save_to is not None:
        p = Path(save_to).with_suffix(".gif")
        ani.save(p, writer="pillow", fps=fps)

    if show:
        return HTML(ani.to_jshtml())


def trajectory_movie(
    y,
    frames=50,
    title="",
    ylabel="",
    xlabel="Time",
    legend=[],
    x=None,
    interval=100,
    ylim=None,
    save_to=None,
):

    y = np.asarray(y)
    if x is None:
        x = np.arange(len(y))

    fig, ax = plt.subplots()
    total = len(x)
    inc = max(total // frames, 1)
    x = x[::inc]
    y = y[::inc]
    if ylim is None:
        ylim = np.array([y.min(), y.max()])
    xlim = [x.min(), x.max()]

    def animate(i):
        ax.cla()
        ax.plot(x[:i], y[:i], marker="o", markevery=[-1])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(legend, loc="lower right")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} t={x[i]:.2f}")

    ani = FuncAnimation(fig, animate, frames=len(x), interval=interval)
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix(".gif")
        ani.save(p, writer="pillow", fps=30)

    return HTML(ani.to_jshtml())


def plot_grid(
    A,
    colorbar=True,
    colorbar_mode="single",
    grid_height=None,
    grid_width=None,
    fig_size=(8, 8),
    cmap="viridis",
    xticks_on=False,
    yticks_on=False,
    aspect="auto",
    space=0.1,
    to_int=False,
    save_to=None,
):
    # a is expected to be an array of images with shape (n, h, w)
    N = A.shape[0]

    # calculate grid dimensions if not provided
    if grid_height is None and grid_width is None:
        grid_width = int(np.ceil(np.sqrt(N)))
        grid_height = int(np.ceil(N / grid_width))
    elif grid_height is None:
        grid_height = int(np.ceil(N / grid_width))
    elif grid_width is None:
        grid_width = int(np.ceil(N / grid_height))

    # create figure
    fig = plt.figure(figsize=fig_size)

    # set up image grid with specified aspect ratio, colorbar mode, and spacing
    cbar_mode = colorbar_mode if colorbar else None
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(grid_height, grid_width),
        axes_pad=space,
        share_all=True,
        cbar_mode=cbar_mode,
        aspect=aspect,
    )

    norm = colors.Normalize(vmin=np.min(A), vmax=np.max(A))
    # plot images
    for i in range(N):
        ax = grid[i]
        im = ax.imshow(A[i], cmap=cmap, aspect="auto", norm=norm)
        if not xticks_on:
            ax.set_xticks([])
        if not yticks_on:
            ax.set_yticks([])

        # add colorbar for each image if needed
        if colorbar and colorbar_mode == "each":
            cbar = ax.cax.colorbar(im)
            # use tick_params to control tick labels
            ax.cax.tick_params(labelleft=True)

    # add single colorbar if needed
    if colorbar and colorbar_mode == "single":
        cbar = grid.cbar_axes[0].colorbar(im)
        # use tick_params to control tick labels
        grid.cbar_axes[0].tick_params()

    if save_to is not None:
        p = Path(save_to)
        plt.savefig(p)

    plt.show()


def plot_grid_movie(
    A,
    t=None,
    titles_x=None,
    titles_y=None,
    suptitle=None,
    suptitle_y=None,
    colorbar=True,
    colorbar_mode="single",
    grid_height=None,
    grid_width=None,
    fig_size=(8, 8),
    cmap="viridis",
    xticks_on=False,
    yticks_on=False,
    aspect="auto",
    space=0.1,
    interval=200,
    save_to=None,
    show=True,
    live_cbar=True,
    c_norm=None,
    seconds=5,
    writer="pillow",
    frames=64,
    to_int=False,
):
    A = np.asarray(A)
    if grid_width is not None and grid_height is not None:
        A = A[: grid_height * grid_width]
    # A is expected to be an array of movies with shape (n, t, h, w)
    t_idx = np.linspace(0, A.shape[1] - 1, frames, dtype=np.int32)
    A = A[:, t_idx]
    n, t = A.shape[:2]

    # Calculate grid dimensions if not provided
    if grid_height is None and grid_width is None:
        grid_width = int(np.ceil(np.sqrt(n)))
        grid_height = int(np.ceil(n / grid_width))
    elif grid_height is None:
        grid_height = int(np.ceil(n / grid_width))
    elif grid_width is None:
        grid_width = int(np.ceil(n / grid_height))

    # Compute vmin and vmax for consistent color scales
    vmin = A.min()
    vmax = A.max()

    # Create figure
    fig = plt.figure(figsize=fig_size)
    if suptitle is not None:
        fig.suptitle(suptitle, y=suptitle_y)

    # Set up image grid with specified aspect ratio, colorbar mode, and spacing
    cbar_mode = colorbar_mode if colorbar else None
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(grid_height, grid_width),
        axes_pad=space,
        share_all=True,
        cbar_mode=cbar_mode,
        aspect=aspect == "auto",
    )

    images = []

    if to_int:
        A = np.clip(A, a_min=0, a_max=255)
        A = np.asarray(A, dtype=np.uint16)

    # Plot initial images
    for i in range(n):
        ax = grid[i]
        if c_norm is not None:
            norm = colors.Normalize(vmin=c_norm[0], vmax=c_norm[1])
        else:
            norm = colors.Normalize(vmin=np.min(A[i, 0]), vmax=np.max(A[i, 0]))
        im = ax.imshow(A[i, 0], cmap=cmap, aspect="auto", norm=norm)
        images.append(im)
        if not xticks_on:
            ax.set_xticks([])
        if not yticks_on:
            ax.set_yticks([])

        # Add colorbar for each image if needed
        if colorbar and colorbar_mode == "each":
            cbar = ax.cax.colorbar(im)
            ax.cax.tick_params(labelleft=True)

        if titles_x is not None and i < grid_width:
            ax.set_title(titles_x[i])

        if titles_y is not None and i % grid_width == 0:
            ax.set_ylabel(titles_y.pop())

    # Add single colorbar if needed
    if colorbar and colorbar_mode == "single":
        cbar = grid.cbar_axes[0].colorbar(im)
        grid.cbar_axes[0].tick_params()

    # Define update function
    def update(frame):
        for i, im in enumerate(images):
            cur = A[i, frame]
            im.set_data(cur)
            if live_cbar:
                vmax = np.max(cur)
                vmin = np.min(cur)
                im.set_clim(vmin, vmax)
        return images

    # Create animation
    ani = FuncAnimation(fig, update, frames=t, interval=interval, blit=False)

    plt.close()
    if save_to is not None:
        p = Path(save_to).with_suffix(".gif")
        fps = t // seconds
        ani.save(
            p,
            writer=writer,
            fps=fps,
        )

    if show:
        return HTML(ani.to_jshtml())
