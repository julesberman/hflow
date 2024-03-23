import hydra

import hflow.io.result as R
from hflow.config import Config
from hflow.data.get import get_data
from hflow.io.save import save_results
from hflow.io.setup import setup
from hflow.net.get import get_colora
from hflow.train.loss import get_loss_fn
from hflow.train.train import train_model


@hydra.main(version_base=None, config_name="default")
def run(cfg: Config) -> None:

    key, data, u_fn, h_fn, psi_theta_init, loss_fn = build(cfg)

    opt_psi_theta = train_model(
        cfg.optimizer, data, loss_fn, psi_theta_init, key)

    save_results(R.RESULT, cfg)


def build(cfg):

    key = setup(cfg)

    data = get_data(cfg.problem, cfg.train_data,
                    cfg.optimizer.batch_size, key)

    u_fn, h_fn, psi_theta_init = get_colora(
        cfg.unet, cfg.hnet, data, key)

    loss_fn = get_loss_fn(cfg.loss, h_fn, u_fn)

    return key, data, u_fn, h_fn, psi_theta_init, loss_fn


if __name__ == "__main__":
    run()
