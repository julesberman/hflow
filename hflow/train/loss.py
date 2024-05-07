
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
    if loss_cfg.loss_fn == 'ov':
        loss_fn = OV_Loss(
            s_fn, noise=noise, sigma=sigma, trace=loss_cfg.trace, t_batches=loss_cfg.t_batches, n_batches=loss_cfg.n_batches)
    elif loss_cfg.loss_fn == 'ncsm':
        sigmas = generate_sigmas(loss_cfg)
        loss_fn = NCSM_Loss(s_fn, sigmas)

    return loss_fn


def OV_Loss(s, noise=0.0, sigma=0.0, return_interior=False, trace='true', t_batches=1, n_batches=1):

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
        # if n_batches > 1:
        #     trace_dds_Vx = batchmap(trace_dds_Vx, n_batches,  argnum=2)
    else:
        trace_dds = tracewrap(jacfwd(s_dx, 1))
        trace_dds_Vx = vmap(trace_dds, (None, 0, None))
        if n_batches > 1:
            trace_dds_Vx = batchmap(trace_dds_Vx, n_batches,  argnum=1)
    if n_batches > 1:
        s_dt_Vx = batchmap(s_dt_Vx, n_batches, argnum=2)
        s_Vx = batchmap(s_Vx, n_batches,  argnum=1)
        s_dx_Vx = batchmap(s_dx_Vx, n_batches,  argnum=1)

    def loss_fn(params, x_t_batch, mu, t_batch, quad_weights, key):
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
        if t_batches > 1:
            interior_loss = batchmap(interior_loss, t_batches)
        interior = interior_loss(xt_tensor)

        if quad_weights is not None:
            loss_interior = (interior * quad_weights).sum()
        else:
            loss_interior = interior.mean()

        loss = (bound.mean() + loss_interior)

        if return_interior:
            return loss, interior

        return loss

    return loss_fn


def NCSM_Loss(s, sigmas):

    def score_match(x, t, sigma, mu, key, params):
        mu_t_sigma = jnp.concatenate([mu, t, sigma])
        seed = jax.random.uniform(key, shape=()) + mu + t + sigma
        seed = jnp.linalg.norm(seed)
        key = jax.random.PRNGKey((seed*1e8).astype(int))
        y = jax.random.normal(key, shape=x.shape)

        x_tilde = x + sigma * y

        l = sigma**2 * 0.5 * \
            jnp.sum((s(mu_t_sigma, x_tilde, params) + (x_tilde - x)/sigma**2)**2)

        return l

    def loss_fn(params, x_t_batch, mu, t_batch, quad_weights, key):

        # remove endpoints from OV loss
        x_t_batch = x_t_batch[1:-1]
        t_batch = t_batch[1:-1]

        T, N, D = x_t_batch.shape
        T, MT = t_batch.shape
        xt_tensor = rearrange(x_t_batch, 'T N D -> T (N D)')
        xt_tensor = jnp.hstack([xt_tensor, t_batch])  # T (2ND + 1)

        def score_match_time_sigma(x_t, sigma):
            x, t = x_t[:-MT], x_t[-MT:]
            x = rearrange(x, '(N D) -> N D', D=D)
            loss = vmap(score_match, (0, None, None, None, None, None))(
                x, t, sigma, mu, key, params)
            return loss.mean()

        score_match_time_sigma = vmap(score_match_time_sigma, (0, None))
        score_match_time_sigma = vmap(score_match_time_sigma, (None, 0))

        return score_match_time_sigma(xt_tensor, sigmas).mean()

    return loss_fn


def generate_sigmas(cfg_loss: Loss):
    start = 1
    end = 1e-2
    L = cfg_loss.L
    sigmas = jnp.asarray([start * (end / start) ** (i / (L - 1))
                          for i in range(L)]).reshape(-1, 1)
    return sigmas
