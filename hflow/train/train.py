import hflow.io.result as R
from hflow.config import Optimizer
from hflow.io.utils import log
from hflow.train.adam import adam_opt


def train_model(opt_cfg: Optimizer, dataset, loss_fn, psi_theta_init, key):

    dataset = iter(dataset)

    # set epochs or iters depending on whats defined, iters overrides epochs
    epochs = opt_cfg.epochs
    bs = dataset.batch_size
    n_batches = dataset.n_batches
    iters = dataset.n_batches * opt_cfg.epochs

    log.info(
        f'adam_opt for n_batches: {n_batches}, iters {iters}, epochs: {epochs}, batch_size: {bs}')

    def args_fn():
        return next(dataset)

    last_psi_theta, opt_psi_theta, loss_history = adam_opt(
        psi_theta_init, loss_fn, args_fn, steps=iters, learning_rate=opt_cfg.lr, verbose=True, scheduler=opt_cfg.scheduler, key=key)

    R.RESULT['last_psi_theta'] = last_psi_theta
    R.RESULT['opt_psi_theta'] = opt_psi_theta
    R.RESULT['loss_history'] = loss_history

    return opt_psi_theta
