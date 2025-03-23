from dataclasses import dataclass, field
from typing import Any, List, Union

from hydra.core.config_store import ConfigStore
from hydra.types import RunMode
from omegaconf import OmegaConf

from hflow.misc.misc import epoch_time, unique_id

# SWEEP = {
#     "problem": "static",
#     "optimizer.iters": "10_000",  # ,50_000',
#     "loss.loss_fn": "ov, dice",
#     "seed": "1,2,3",
#     "sample.scheme_t": "simp,rand",
#     "loss.sigma": "0.0",
# }

SWEEP = {
    "problem": "ip",
    "optimizer.iters": "10_000, 25_000, 50_000",  
    "loss.loss_fn": "dice",
    "seed": "1,2,3",
    "sample.scheme_t": "rand",
    "loss.sigma": "0.0",
    "x64": "True",
    "sample.bs_n": "512",
    "net.width": "128",
    "net.cond_in": "append",
    "data.normalize": "False"
}

SLURM_CONFIG = {
    "timeout_min": 60*3,
    "cpus_per_task": 4,
    "mem_gb": 200,
    "gpus_per_node": 1,
    "gres": "gpu",
    "account": "extremedata",
}


@dataclass
class Network:
    width: int = 64
    depth: int = 6
    cond_features: List[int] = field(default_factory=lambda: [32] * 3)
    activation: str = "gelu"
    bias: bool = True
    fix_u: Union[bool, None] = None
    homo: bool = False
    cond_in: str = 'film'


@dataclass
class Optimizer:
    lr: float = 5e-4
    iters: int = 25_000
    scheduler: bool = True
    optimizer: str = "adam"
    save_params_history: bool = False


@dataclass
class Data:
    ode: str = "euler"
    dt: float = 5e-3
    t_end: float = 10
    n_samples: int = 10_000
    normalize: bool = True
    save: bool = False
    load: bool = False
    dim: Union[int, None] = None
    omega: float = 8.0


@dataclass
class Loss:
    loss_fn: str = "ov"
    sigma: float = 1e-1
    log: bool = False
    trace: str = "normal"
    L: int = 10
    T: int = 100
    alpha: float = 0.0
    alpha_quant: str = "s"  # 's' or 's_t


@dataclass
class Sample:
    bs_n: int = 256
    bs_t: int = 256
    scheme_t: str = "simp"
    scheme_n: str = "rand"
    scheme_w: str = "normal"
    all_mu: bool = False


@dataclass
class Test:
    run: bool = True
    dt: float = 1e-3
    t_samples: Union[int, None] = 128
    n_samples: int = 20_000
    n_plot_samples: int = 2000
    plot_particles: bool = False
    plot_hist: bool = False
    plot_func: bool = False
    w_eps: float = 0.01
    noise_type: str = "sde"
    electric: bool = False
    save_sol: bool = False
    mean: bool = False
    wass: bool = False
    analytic: bool = False


@dataclass
class Config:

    problem: str

    net: Network = field(default_factory=Network)
    # unet: Network = field(default_factory=Network)
    # hnet: Network = field(default_factory=lambda: Network(width=15, layers=["D"] * 3))
    optimizer: Optimizer = field(default_factory=Optimizer)
    data: Data = field(default_factory=Data)

    loss: Loss = field(default_factory=Loss)

    test: Test = field(default_factory=Test)
    sample: Sample = field(default_factory=Sample)

    # misc
    name: str = field(default_factory=lambda: f"{unique_id(4)}_{epoch_time(2)}")
    x64: bool = True  # whether to use 64 bit precision in jax

    platform: Union[str, None] = None  # gpu or cpu, None will let jax default
    # output_dir: str = './results/${hydra.job.name}'  # where to save results, if None nothing is saved

    seed: int = 1
    debug_nans: bool = False  # whether to debug nans
    advanced_flags: bool = (
        True  # set advanced flags https://jax.readthedocs.io/en/latest/gpu_performance_tips.html#nccl-flags
    )
    # optional info about details of the experiment
    info: Union[str, None] = None

    # hydra config configuration
    hydra: Any = field(default_factory=lambda: hydra_config)
    defaults: List[Any] = field(default_factory=lambda: defaults)


##########################
## hydra settings stuff ##
##########################
defaults = [
    # https://hydra.cc/docs/tutorials/structured_config/defaults/
    # "_self_",
    {"override hydra/launcher": "submitit_slurm"},
    # add color logging
    {"override hydra/job_logging": "colorlog"},
    {"override hydra/hydra_logging": "colorlog"},
]


# def get_mode():
#     if len(SWEEP.keys()) > 0:
#         return RunMode.MULTIRUN
#     return RunMode.RUN


hydra_config = {
    # sets the out dir from config.problem and id
    "run": {"dir": "results/${problem}/single/${name}"},
    "sweep": {"dir": "results/${problem}/multi/${name}"},
    # "mode": get_mode(),
    "sweeper": {"params": {**SWEEP}},
    # https://hydra.cc/docs/1.2/plugins/submitit_launcher/
    "launcher": {**SLURM_CONFIG},
    # sets logging config
    "job_logging": {
        "formatters": {"colorlog": {"format": "[%(levelname)s] - %(message)s"}}
    },
    "job": {"env_set": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}},
}
import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)

##################################
## problem wise default configs ##
##################################


cs = ConfigStore.instance()
cs.store(name="default", node=Config)

two_config = Config(
    problem="vtwo",
    loss=Loss(sigma=5e-2),
    data=Data(t_end=40, n_samples=25_000, dt=1e-2),
    test=Test(plot_hist=True, electric=True, wass=True, n_samples=25_000),
)

bump_config = Config(
    problem="vbump",
    loss=Loss(sigma=5e-2),
    data=Data(t_end=40, n_samples=25_000, dt=1e-2),
    test=Test(plot_hist=True, electric=True, wass=True, n_samples=25_000),
)

bi_config = Config(
    problem="bi",
    data=Data(t_end=12, dt=5e-3, n_samples=25_000),
    test=Test(plot_particles=True, mean=True, wass=True),
)

trap_config = Config(
    problem="trap",
    data=Data(t_end=2, dim=100, n_samples=5000, dt=7.5e-3),
    sample=Sample(bs_n=256, bs_t=256),
    test=Test(plot_particles=True, mean=True),
)

mdyn_config = Config(
    problem="mdyn",
    data=Data(t_end=1, dim=2, n_samples=10_000, dt=2e-3),
    test=Test(plot_particles=True, wass=True),
)

lz9_config = Config(
    problem="lz9",
    data=Data(t_end=20, n_samples=25_000, dt=1e-2),
    loss=Loss(sigma=1e-1),
    test=Test(
        plot_particles=True, wass=True, mean=True, n_samples=25_000, t_samples=32
    ),
)

v6_config = Config(
    problem="v6",
    loss=Loss(sigma=5e-2),
    data=Data(n_samples=25_000, t_end=6),
    test=Test(plot_hist=True, wass=True, n_samples=25_000, electric=True),
)

lin_config = Config(
    problem="lin",
    data=Data(t_end=1, dt=2e-3, n_samples=10_000, omega=8.0),
    optimizer=Optimizer(iters=10_000),
    loss=Loss(sigma=1e-2),
    test=Test(plot_particles=True, mean=True, wass=False, analytic=True),
)


static_config = Config(
    problem="static",
    data=Data(t_end=1, dt=(1/512), n_samples=10_000
    ),
    test=Test(plot_particles=True, mean=True, wass=True),
    net=Network(width=32, depth=3, cond_in='append'),
    loss=Loss(sigma=0.0, loss_fn='dice'),
    optimizer=Optimizer(lr=1e-3, iters=25_000),
    sample=Sample(bs_n=128, bs_t=128, scheme_t='rand')
)

ip_config = Config(
    problem="ip",
    data=Data(n_samples=10_000),
    test=Test(plot_particles=True, mean=True, wass=True, plot_hist=True),
    loss=Loss(sigma=0.0, loss_fn='dice'),
    sample=Sample(bs_n=512, bs_t=256, scheme_t='rand'),
    net=Network(width=256, depth=6, cond_in='film'),
)

cs.store(name="lz9", node=lz9_config)
cs.store(name="mdyn", node=mdyn_config)
cs.store(name="trap", node=trap_config)
cs.store(name="bi", node=bi_config)
cs.store(name="vtwo", node=two_config)
cs.store(name="vbump", node=bump_config)
cs.store(name="v6", node=v6_config)
cs.store(name="lin", node=lin_config)
cs.store(name="static", node=static_config)
cs.store(name="ip", node=ip_config)