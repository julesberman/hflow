from pathlib import Path

import matplotlib
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pgf import FigureCanvasPgf

PLOT_PATH = Path('../plots/')


def reset_style():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def set_style(latex=True):
    # reset to default
    reset_style()

    # style
    matplotlib.style.use('seaborn-v0_8-whitegrid')
    matplotlib.rcParams["legend.frameon"] = True
    matplotlib.rcParams["lines.linewidth"] = 1.2
    matplotlib.rcParams["axes.linewidth"] = 0.8
    matplotlib.rcParams["axes.edgecolor"] = 'black'
    matplotlib.rcParams["ytick.major.size"] = 2
    matplotlib.rcParams["xtick.major.size"] = 2

    if latex:
        matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            'font.size': 10,
        })

    # plt.gcf().set_tight_layout(True)
    # matplotlib.rcParams["font.size"] = 19
    # matplotlib.rcParams["legend.fontsize"] = 19
    # matplotlib.rcParams["ytick.labelsize"] = 19
    # matplotlib.rcParams["xtick.labelsize"] = 19
    # matplotlib.rcParams["axes.labelsize"] = 27


def save_show(path, save=True, show=True, format='pgf'):

    if save:
        plt.savefig(path, format=format, bbox_inches='tight', pad_inches=0.01)
    if show:
        plt.show()
