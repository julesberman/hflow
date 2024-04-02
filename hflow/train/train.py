import hflow.io.result as R
from hflow.config import Optimizer
from hflow.io.utils import log
from hflow.train.adam import adam_opt


def train_model(opt_cfg: Optimizer, arg_fn, loss_fn, psi_theta_init, key):

    iters = opt_cfg.iters
    # log.info(
    #     f'adam_opt for n_batches: {n_batches}, iters {iters}, epochs: {epochs}, batch_size: {bs}')

    last_params, opt_params, loss_history = adam_opt(
        psi_theta_init, loss_fn, arg_fn, steps=iters, learning_rate=opt_cfg.lr, verbose=True, scheduler=opt_cfg.scheduler, key=key)

    R.RESULT['last_params'] = last_params
    R.RESULT['opt_params'] = opt_params
    R.RESULT['loss_history'] = loss_history

    return last_params
