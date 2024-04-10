
from dataclasses import dataclass, field
from typing import Any, List, Union

from hydra.core.config_store import ConfigStore
from hydra.types import RunMode
from omegaconf import OmegaConf

from hflow.misc.misc import epoch_time, unique_id

# sweep configurationm, if empty will not sweep
SWEEP = {}
SWEEP = {
    'optimizer.iters': '5_000,25_000,100_000',
    'unet.width': '32,64,128',
    # 'sample.bs_t': '8,16,32,64,128',
    'sample.scheme_t': 'rand,gauss'
}

SLURM_CONFIG = {
    'timeout_min': 60*2,
    'cpus_per_task': 8,
    'mem_gb': 25,
    # 'gpus_per_node': 1,
    'gres': 'gpu'
}


@dataclass
class Network:
    model: str = 'dnn'
    width: int = 32
    layers: List[str] = field(default_factory=lambda: [
                              'C']*7)  # ['P',*['C']*7])
    activation: str = 'swish'
    rank: int = 3
    full: bool = True
    bias: bool = True
    last_activation: Union[str, None] = None


@dataclass
class Optimizer:
    lr: float = 5e-3
    iters: int = 2_000
    scheduler: bool = True
    optimizer: str = 'adamw'


@dataclass
class Data:
    ode: str = 'euler'
    dt: float = 5e-3
    t_end: int = 10
    n_samples: int = 20_000
    normalize: bool = True


@dataclass
class Loss:
    loss_fn: str = 'am'
    noise: float = 0.0
    sigma: float = 1e-1


@dataclass
class Sample:
    bs_n: int = 128
    bs_t: int = 128
    scheme_t: str = 'gauss'
    scheme_n: str = 'rand'


@dataclass
class Test:
    run: bool = True
    dt: float = 1e-3
    n_samples: int = 25_000
    plot_samples: int = 2000
    plot: bool = True


@dataclass
class Config:

    problem: str

    unet: Network = field(default_factory=Network)
    hnet: Network = field(
        default_factory=lambda: Network(width=15, layers=['D']*3))
    optimizer: Optimizer = field(default_factory=Optimizer)
    data: Data = field(default_factory=Data)

    loss: Loss = field(default_factory=Loss)

    test: Test = field(default_factory=Test)
    sample: Sample = field(default_factory=Sample)

    # misc
    name: str = field(default_factory=lambda: epoch_time(2))
    x64: bool = False  # whether to use 64 bit precision in jax
    platform: Union[str, None] = None  # gpu or cpu, None will let jax default
    # output_dir: str = './results/${hydra.job.name}'  # where to save results, if None nothing is saved

    seed: int = 1
    debug_nans: bool = False  # weather to debug nans
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
    {"override hydra/hydra_logging": "colorlog"}
]


# def get_mode():
#     if len(SWEEP.keys()) > 0:
#         return RunMode.MULTIRUN
#     return RunMode.RUN


hydra_config = {
    # sets the out dir from config.problem and id
    "run": {
        "dir": "results/${problem}/single/${name}"
    },
    "sweep": {
        "dir": "results/${problem}/multi/${name}"
    },

    # "mode": get_mode(),
    "sweeper": {
        "params": {
            **SWEEP
        }
    },
    # https://hydra.cc/docs/1.2/plugins/submitit_launcher/
    "launcher": {
        **SLURM_CONFIG
    },
    # sets logging config
    "job_logging": {
        "formatters": {
            "colorlog": {
                "format": '[%(levelname)s] - %(message)s'
            }
        }
    },
    "job": {
        "env_set": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
        }
    }
}


##################################
## problem wise default configs ##
##################################


cs = ConfigStore.instance()
cs.store(name="default", node=Config)

vlasov_config = Config(problem='vlasov',
                       data=Data(t_end=30),
                       unet=Network(layers=['P', *['C']*7]))


osc_config = Config(problem='osc',
                    data=Data(t_end=15))


sburgers_config = Config(problem='sburgers',
                         data=Data(t_end=4, n_samples=512),
                         test=Test(n_samples=25),
                         unet=Network(width=64))


cs.store(name="osc", node=osc_config)
cs.store(name="vlasov", node=vlasov_config)
cs.store(name="sburgers", node=sburgers_config)
