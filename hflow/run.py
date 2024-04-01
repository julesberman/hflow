import hydra

import hflow.io.result as R
from hflow.config import Config
from hflow.data.get import get_data
from hflow.io.save import save_results
from hflow.io.setup import setup
from hflow.net.get import get_network
from hflow.train.loss import get_loss_fn
from hflow.train.sample import get_arg_fn
from hflow.train.train import train_model


@hydra.main(version_base=None, config_name="default")
def run(cfg: Config) -> None:

    key, data, loss_fn, arg_fn, s_fn, params_init = build(cfg)

    opt_params = train_model(
        cfg.optimizer, arg_fn, loss_fn, params_init, key)

    save_results(R.RESULT, cfg)


def build(cfg: Config):

    key = setup(cfg)

    data = get_data(cfg.problem, cfg.data)

    s_fn, params_init = get_network(cfg, data, key)

    arg_fn = get_arg_fn(cfg.sample, data)
    loss_fn = get_loss_fn(cfg.loss, s_fn)

    return key, data, loss_fn, arg_fn, s_fn, params_init


if __name__ == "__main__":
    run()
