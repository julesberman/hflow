from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import grad, jacfwd, jacrev, jit, jvp, vmap
from jax.experimental.host_callback import id_print

from hflow.config import Loss
from hflow.misc.jax import get_rand_idx, hess_trace_estimator


def get_loss_fn(loss_cfg: Loss, s_fn):

    noise = loss_cfg.noise
    sigma = loss_cfg.sigma
    match loss_cfg.loss_fn:

        case 'am':
            loss_fn = Action_Match(s_fn, noise=noise, sigma=sigma)

    return loss_fn


def Action_Match(s, noise=0.0, sigma=0.0):

    s_dt = jacrev(s, 0)
    s_dt_Vx = vmap(s_dt, (None, 0, None))
    s_Vx = vmap(s, in_axes=(None, 0, None))
    s_dx = jacrev(s, 1)
    s_dx_Vx = vmap(s_dx, in_axes=(None, 0, None))
    s_ddx_Vx = vmap(jacfwd(s_dx, 1), (None, 0, None))
    s_dt_dx = jacrev(s, (0, 1))
    s_dt_dx_Vx = vmap(s_dt_dx, in_axes=(None, 0, None))
    trace_dds = hess_trace_estimator(s, argnum=1)
    trace_dds_Vx = vmap(trace_dds, (None, None, 0, None))

    def am_loss(psi_theta, x_t_batch, t_batch, key):
        x_t_batch += jax.random.normal(key, x_t_batch.shape)*noise

        T, N, D = x_t_batch.shape
        T, MT = t_batch.shape
        bound = s_Vx(t_batch[0], x_t_batch[0], psi_theta) - \
            s_Vx(t_batch[-1], x_t_batch[-1], psi_theta)

        xt_tensor = rearrange(x_t_batch, 'T N D -> T (N D)')
        xt_tensor = jnp.hstack([xt_tensor, t_batch])

        def interior_loss(xt_tensor):
            x_batch, t = xt_tensor[:-MT], xt_tensor[-MT:]
            x_batch = rearrange(x_batch, '(N D) -> N D', D=D)

            ut = s_dt_Vx(t, x_batch, psi_theta)
            # entropic
            if sigma > 0.0:
                gu, trace_ets = trace_dds_Vx(key, t, x_batch, psi_theta)
                ent = trace_ets*sigma  # **2*0.5
            else:
                gu = s_dx_Vx(t, x_batch, psi_theta)
                ent = 0.0

            gu = 0.5*jnp.sum(gu**2, axis=1)

            return (ut+gu+ent).mean()

        interior = vmap(interior_loss)(xt_tensor)

        loss = (bound.mean() + interior.mean())

        return loss

    return am_loss
