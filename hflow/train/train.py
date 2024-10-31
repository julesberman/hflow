import hflow.io.result as R
from hflow.config import Optimizer
from hflow.io.utils import log
from hflow.train.adam import adam_opt

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def train_model(opt_cfg: Optimizer, arg_fn, loss_fn, psi_theta_init, key):

    iters = opt_cfg.iters

    last_params, opt_params, loss_history, param_history = adam_opt(
        psi_theta_init,
        loss_fn,
        arg_fn,
        steps=iters,
        learning_rate=opt_cfg.lr,
        verbose=True,
        scheduler=opt_cfg.scheduler,
        key=key,
        return_params=True,
    )

    R.RESULT["last_params"] = last_params
    R.RESULT["opt_params"] = opt_params
    R.RESULT["loss_history"] = loss_history
    if opt_cfg.save_params_history:
        R.RESULT["param_history"] = param_history

    return last_params
