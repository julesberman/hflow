
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import grad, jacfwd, jacrev, jit, jvp, vmap
from jax.experimental.host_callback import id_print

from hflow.config import Data, Loss
from hflow.misc.jax import batchmap, hess_trace_estimator, tracewrap


def get_loss_fn(loss_cfg: Loss, data_cfg: Data, s_fn):

    noise = loss_cfg.noise
    sigma = loss_cfg.sigma
    if loss_cfg.loss_fn == 'am':
        loss_fn = OV_Loss(
            s_fn, noise=noise, sigma=sigma, trace=loss_cfg.trace, batches=data_cfg.batches)

    return loss_fn


def OV_Loss(s, noise=0.0, sigma=0.0, return_interior=False, trace='true', batches=1):

    def s_sep(mu, t, x, params):
        mu_t = jnp.concatenate([mu, t])
        return s(mu_t, x, params)

    s_dt = jacrev(s_sep, 1)
    s_dt_Vx = vmap(s_dt, (None, None, 0, None))
    s_Vx = vmap(s, in_axes=(None, 0, None))
    s_dx = jacrev(s, 1)
    s_dx_Vx = vmap(s_dx, in_axes=(None, 0, None))

    if trace == 'hutch':
        trace_dds = hess_trace_estimator(s, argnum=1)
        trace_dds_Vx = vmap(trace_dds, (None, None, 0, None))
    else:
        trace_dds = tracewrap(jacfwd(s_dx, 1))
        trace_dds_Vx = vmap(trace_dds, (None, 0, None))

    def am_loss(params, x_t_batch, mu, t_batch, quad_weights, key):
        x_t_batch += jax.random.normal(key, x_t_batch.shape)*noise

        t_batch = jnp.hstack([jnp.ones((len(t_batch), len(mu))) * mu, t_batch])

        T, N, D = x_t_batch.shape
        T, MT = t_batch.shape
        bound = s_Vx(t_batch[0], x_t_batch[0], params) - \
            s_Vx(t_batch[-1], x_t_batch[-1], params)

        x_t_batch = x_t_batch[1:-1]
        t_batch = t_batch[1:-1]

        xt_tensor = rearrange(x_t_batch, 'T N D -> T (N D)')
        xt_tensor = jnp.hstack([xt_tensor, t_batch])

        def interior_loss(xt_tensor):
            x_batch, mu_t = xt_tensor[:-MT], xt_tensor[-MT:]
            x_batch = rearrange(x_batch, '(N D) -> N D', D=D)

            mu, t = mu_t[:-1], mu_t[-1:]
            ut = s_dt_Vx(mu, t, x_batch, params)
            ut = jnp.squeeze(ut)

            # entropic
            if sigma > 0.0:
                if trace == 'hutch':
                    gu, tra = trace_dds_Vx(key, mu_t, x_batch, params)
                else:
                    tra = trace_dds_Vx(mu_t, x_batch, params)
                    gu = s_dx_Vx(mu_t, x_batch, params)
                ent = tra*sigma**2*0.5
            else:
                gu = s_dx_Vx(mu_t, x_batch, params)
                ent = 0.0

            gu = 0.5*jnp.sum(gu**2, axis=1)

            return (ut+gu+ent).mean()

        interior_loss = vmap(interior_loss)
        if batches > 1:
            interior_loss = batchmap(interior_loss, batches)
        interior = interior_loss(xt_tensor)

        if quad_weights is not None:
            loss_interior = (interior * quad_weights).sum()
        else:
            loss_interior = interior.mean()

        loss = (bound.mean() + loss_interior)

        if return_interior:
            return loss, interior

        return loss

    return am_loss
