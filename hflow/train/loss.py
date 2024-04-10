
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import grad, jacfwd, jacrev, jit, jvp, vmap
from jax.experimental.host_callback import id_print

from hflow.config import Loss
from hflow.misc.jax import hess_trace_estimator


def get_loss_fn(loss_cfg: Loss, s_fn):

    noise = loss_cfg.noise
    sigma = loss_cfg.sigma
    if loss_cfg.loss_fn == 'am':
        loss_fn = Action_Match(s_fn, noise=noise, sigma=sigma)

    return loss_fn


def Action_Match(s, noise=0.0, sigma=0.0, return_interior=False):

    def s_sep(mu, t, x, params):
        mu_t = jnp.concatenate([mu, t])
        return s(mu_t, x, params)

    s_dt = jacrev(s_sep, 1)
    s_dt_Vx = vmap(s_dt, (None, None, 0, None))
    s_Vx = vmap(s, in_axes=(None, 0, None))
    s_dx = jacrev(s, 1)
    s_dx_Vx = vmap(s_dx, in_axes=(None, 0, None))

    trace_dds = hess_trace_estimator(s, argnum=1)
    trace_dds_Vx = vmap(trace_dds, (None, None, 0, None))

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

            mu, t = mu_t[:1], mu_t[1:]
            ut = s_dt_Vx(mu, t, x_batch, params)
            ut = jnp.squeeze(ut)

            # entropic
            if sigma > 0.0:
                gu, trace_ets = trace_dds_Vx(key, mu_t, x_batch, params)
                ent = trace_ets*sigma**2*0.5
            else:
                gu = s_dx_Vx(mu_t, x_batch, params)
                ent = 0.0

            gu = 0.5*jnp.sum(gu**2, axis=1)

            return (ut+gu+ent).mean()

        interior = vmap(interior_loss)(xt_tensor)

        if quad_weights is not None:
            interior_w = interior * quad_weights

        loss = (bound.mean() + interior_w.sum())

        if return_interior:
            return loss, interior

        return loss

    return am_loss
