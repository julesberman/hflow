from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import grad, jacfwd, jacrev, jit, jvp, vmap
from jax.flatten_util import ravel_pytree

from hflow.config import Loss, Sample
from hflow.misc.jax import hess_trace_estimator, meanvmap, tracewrap


def get_loss_fn(loss_cfg: Loss, sample_cfg: Sample, s_fn):

    if loss_cfg.loss_fn == "ov":
        loss_fn = OV_Loss(
            s_fn,
            sigma=loss_cfg.sigma,
            alpha=loss_cfg.alpha,
            alpha_quant=loss_cfg.alpha_quant,
        )
    # elif loss_cfg.loss_fn == "ov_old":
    #     loss_fn = OV_Loss_old(s_fn, sigma=loss_cfg.sigma, trace=loss_cfg.trace)
    elif loss_cfg.loss_fn == "dice":
        loss_fn = DICE_Loss(
            s_fn,
            sigma=loss_cfg.sigma,
            alpha=loss_cfg.alpha,
            alpha_quant=loss_cfg.alpha_quant,
        )
    elif loss_cfg.loss_fn == "ncsm":
        sigmas = generate_sigmas(loss_cfg.L)
        loss_fn = NCSM_Loss(s_fn, sigmas)
    elif loss_cfg.loss_fn == "cfm":
        loss_fn = CFM_Loss(s_fn)
    elif loss_cfg.loss_fn == "si":
        loss_fn = SI_Loss(s_fn, loss_cfg.sigma)

    return loss_fn


def DICE_Loss(s, sigma=0.0, alpha=0.0, alpha_quant=None):

    def s_sep(mu, x, t, params):
        mu_t = jnp.concatenate([mu, t])
        return s(mu_t, x, params)

    s_Ex = meanvmap(s_sep, in_axes=(None, 0, None, None))
    s_Ex_Vt = vmap(s_Ex, in_axes=(None, 0, 0, None))

    s_dx = jacrev(s_sep, 1)
    s_dx_sq = lambda *args: jnp.sum(s_dx(*args) ** 2)

    sq_dx_Ex = meanvmap(s_dx_sq, in_axes=(None, 0, None, None))
    sq_dx_Ex_Vt = vmap(sq_dx_Ex, in_axes=(None, 0, 0, None))

    s_dt = jacrev(s_sep, 2)
    dt_Ex = meanvmap(s_dt, in_axes=(None, 0, None, None))
    dt_Ex_Vt = vmap(dt_Ex, in_axes=(None, 0, 0, None))

    # if trace == 'hutch':
    #     laplace = hess_trace_estimator(s_sep, argnum=1)
    #     laplace_Ex = meanvmap(laplace, in_axes=(None, None, 0, None, None))
    #     hutch_laplace_Ex_Vt = vmap(laplace_Ex, in_axes=(None, None, 0, 0, None))
    # else:
    laplace = tracewrap(jacfwd(s_dx, 1))
    laplace_Ex = meanvmap(laplace, in_axes=(None, 0, None, None))
    laplace_Ex_Vt = vmap(laplace_Ex, in_axes=(None, 0, 0, None))

    def loss_fn(params, x_t_batch, mu, t_batch, quad_weights, key):

        # remove endpoints from OV loss
        x_t_batch = x_t_batch[1:-1]
        t_batch = t_batch[1:-1]

        dists = t_batch[1:] - t_batch[:-1]
        dists = jnp.squeeze(dists) * 0.5

        s_XT_p1 = s_Ex_Vt(mu, x_t_batch[1:], t_batch[:-1], params)
        s_XT_m1 = s_Ex_Vt(mu, x_t_batch[:-1], t_batch[1:], params)
        s_XT_a = s_Ex_Vt(mu, x_t_batch, t_batch, params)

        diff = s_XT_p1 - s_XT_m1 + s_XT_a[1:] - s_XT_a[:-1]
        diff = jnp.sum(diff)

        sdx_XT = sq_dx_Ex_Vt(mu, x_t_batch, t_batch, params)  # T
        grad = sdx_XT[:-1] + sdx_XT[1:]
        grad = jnp.sum(grad * dists)

        loss = (grad - diff) * 0.5

        if sigma > 0.0:
            lap = laplace_Ex_Vt(mu, x_t_batch, t_batch, params)
            lap = lap[:-1] + lap[1:]
            lap = jnp.sum(lap * dists)
            lap = 0.5 * sigma**2 * lap
            loss += lap

        if alpha > 0.0:
            reg = 0.0
            if alpha_quant == "s":
                reg = s_Ex_Vt(mu, x_t_batch, t_batch, params) ** 2
            if alpha_quant == "s_t":
                dt = dt_Ex_Vt(mu, x_t_batch, t_batch, params)
                dt = jnp.squeeze(dt)
                reg = dt**2
            reg = reg[:-1] + reg[1:]
            loss += jnp.sum(reg * dists) * alpha

        return loss

    return loss_fn


def OV_Loss(s_combine, sigma=0.0, alpha=0.0, alpha_quant=None):

    def s_fn(mu, x, t, params):
        mu_t = jnp.concatenate([mu, t])
        return s_combine(mu_t, x, params)

    s_Ex = meanvmap(s_fn, in_axes=(None, 0, None, None))
    s_Ex_Vt = vmap(s_Ex, in_axes=(None, 0, 0, None))

    s_dx = jacrev(s_fn, 1)
    s_dx_norm = lambda *args: jnp.sum(s_dx(*args) ** 2)
    s_dx_norm_Ex = meanvmap(s_dx_norm, in_axes=(None, 0, None, None))
    s_dx_norm_Ex_Vt = vmap(s_dx_norm_Ex, in_axes=(None, 0, 0, None))

    s_dt = jacrev(s_fn, 2)
    dt_Ex = meanvmap(s_dt, in_axes=(None, 0, None, None))
    dt_Ex_Vt = vmap(dt_Ex, in_axes=(None, 0, 0, None))

    laplace = tracewrap(jacfwd(s_dx, 1))
    laplace_Ex = meanvmap(laplace, in_axes=(None, 0, None, None))
    laplace_Ex_Vt = vmap(laplace_Ex, in_axes=(None, 0, 0, None))

    epsilon = sigma

    def loss_fn(params, X_batch, mu, t_batch, quad_weights, key):
        boundary_term = s_Ex(mu, X_batch[0], t_batch[0], params) - s_Ex(
            mu, X_batch[-1], t_batch[-1], params
        )

        X_batch = X_batch[1:-1]
        t_batch = t_batch[1:-1]

        grad = s_dx_norm_Ex_Vt(mu, X_batch, t_batch, params)
        dt = dt_Ex_Vt(mu, X_batch, t_batch, params)
        dt = jnp.squeeze(dt)

        if epsilon > 0.0:
            lap = laplace_Ex_Vt(mu, X_batch, t_batch, params)
        else:
            lap = 0.0

        interior = 0.5 * grad + dt + epsilon**2 * 0.5 * lap

        if alpha > 0.0:
            reg = 0.0
            if alpha_quant == "s":
                reg = s_Ex_Vt(mu, X_batch, t_batch, params) ** 2
            if alpha_quant == "s_t":
                reg = dt**2
            if alpha_quant == "p":
                params_flat = ravel_pytree(params)[0]
                reg = jnp.linalg.norm(params_flat)
            interior += reg * alpha

        if quad_weights is not None:
            interior_loss = (interior * quad_weights).sum()
        else:
            interior_loss = interior.mean()

        loss = interior_loss + boundary_term

        return loss

    return loss_fn


def NCSM_Loss(s, sigmas, t_batches=1):

    def score_match(x, t, sigma, mu, key, params):
        mu_t_sigma = jnp.concatenate([mu, t, sigma])
        seed = jax.random.uniform(key, shape=()) + mu + t + sigma
        seed = jnp.linalg.norm(seed)
        key = jax.random.PRNGKey((seed * 1e8).astype(int))
        y = jax.random.normal(key, shape=x.shape)

        x_tilde = x + sigma * y

        l = (
            sigma**2
            * 0.5
            * jnp.sum((s(mu_t_sigma, x_tilde, params) + (x_tilde - x) / sigma**2) ** 2)
        )

        return l

    def loss_fn(params, x_t_batch, mu, t_batch, quad_weights, key):

        # remove endpoints from OV loss
        x_t_batch = x_t_batch[1:-1]
        t_batch = t_batch[1:-1]

        T, N, D = x_t_batch.shape
        T, MT = t_batch.shape
        xt_tensor = rearrange(x_t_batch, "T N D -> T (N D)")
        xt_tensor = jnp.hstack([xt_tensor, t_batch])  # T (2ND + 1)

        def score_match_time_sigma(x_t, sigma):
            x, t = x_t[:-MT], x_t[-MT:]
            x = rearrange(x, "(N D) -> N D", D=D)
            loss = vmap(score_match, (0, None, None, None, None, None))(
                x, t, sigma, mu, key, params
            )
            return loss.mean()

        score_match_time_sigma = vmap(score_match_time_sigma, (0, None))
        score_match_time_sigma = vmap(score_match_time_sigma, (None, 0))

        return score_match_time_sigma(xt_tensor, sigmas).mean()

    return loss_fn


def generate_sigmas(L, start=1, end=1e-2):

    sigmas = jnp.asarray(
        [start * (end / start) ** (i / (L - 1)) for i in range(L)]
    ).reshape(-1, 1)
    return sigmas


def Label_Loss(s_fn, t_batches=1):

    def loss_fn(params, x_t_batch, mu, t_batch, quad_weights, key):

        # remove endpoints from OV loss
        x_t_batch = x_t_batch[1:-1]
        t_batch = t_batch[1:-1]

        T, N, D = x_t_batch.shape
        T, MT = t_batch.shape

        vec = x_t_batch

    return loss_fn


def CFM_Loss(s_fn, t_batches=1):

    def loss_fn(params, x_t_batch, mu, t_batch, quad_weights, key):

        x0 = x_t_batch[0]

        # remove endpoints from OV loss
        x_t_batch = x_t_batch[1:-1]
        t_batch = t_batch[1:-1]

        T, N, D = x_t_batch.shape
        T, MT = t_batch.shape
        xt_tensor = rearrange(x_t_batch, "T N D -> T (N D)")
        xt_tensor = jnp.hstack([xt_tensor, t_batch])  # T (2ND + 1)

        s_Vx = vmap(s_fn, (None, 0, None))

        def flow_match(xt, t, tau, mu, params):
            mu_t_tau = jnp.concatenate([mu, t, tau])
            x_tau = tau * xt + (1 - (1 - 0.0) * tau) * x0
            interior = s_Vx(mu_t_tau, x_tau, params) - xt + (1 - 0.0) * x0
            l = 0.5 * jnp.sum(interior**2, axis=-1)
            return l.mean()

        def flow_match_time(x_t, tau):
            x, t = x_t[:-MT], x_t[-MT:]
            x = rearrange(x, "(N D) -> N D", D=D)
            loss = flow_match(x, t, tau, mu, params)
            return loss.mean()

        flow_match_time = vmap(flow_match_time, (0, None))
        flow_match_time = vmap(flow_match_time, (None, 0))

        bs_tau = 32
        taus = jax.random.uniform(key, (bs_tau, 1))

        return flow_match_time(xt_tensor, taus).mean()

    return loss_fn


def SI_Loss(s_fn, sigma):

    sigma = 0.0

    def loss_fn(params, x_t_batch, mu, t_batch, quad_weights, key):

        x0 = x_t_batch[0]

        # remove endpoints from OV loss
        x_t_batch = x_t_batch[1:-1]
        t_batch = t_batch[1:-1]

        T, N, D = x_t_batch.shape
        T, MT = t_batch.shape
        xt_tensor = rearrange(x_t_batch, "T N D -> T (N D)")
        xt_tensor = jnp.hstack([xt_tensor, t_batch])  # T (2ND + 1)

        s_Vx = vmap(s_fn, (None, 0, None))

        def flow_match(xt, t, tau, mu, params):
            mu_t_tau = jnp.concatenate([mu, t, tau])
            # x_tau = tau * xt + (1 - (1 - 0.0)*tau) * x0
            seed = jax.random.uniform(key, shape=()) + mu + t + tau
            seed = jnp.linalg.norm(seed)
            keyr = jax.random.PRNGKey((seed * 1e8).astype(int))
            r = jax.random.normal(keyr, shape=xt.shape)

            a = sigma**2 / 2
            alpha = jnp.cos(jnp.pi * tau / 2)
            beta = jnp.sin(jnp.pi * tau / 2)
            gamma = jnp.sqrt(a * tau * (1 - tau))
            x_tau = beta * xt + alpha * x0 + gamma * r

            u = (
                -jnp.pi / 2 * beta * x0
                + jnp.pi / 2 * alpha * xt
                + 1 / 2 * 1 / gamma * (a - 2 * a * tau) * r
            )
            interior = s_Vx(mu_t_tau, x_tau, params) - u

            l = 0.5 * jnp.sum(interior**2, axis=-1)
            return l.mean()

        def flow_match_time(x_t, tau):
            x, t = x_t[:-MT], x_t[-MT:]
            x = rearrange(x, "(N D) -> N D", D=D)
            loss = flow_match(x, t, tau, mu, params)
            return loss.mean()

        flow_match_time = vmap(flow_match_time, (0, None))
        flow_match_time = vmap(flow_match_time, (None, 0))

        bs_tau = 32
        taus = jax.random.uniform(key, (bs_tau, 1))

        return flow_match_time(xt_tensor, taus).mean()

    return loss_fn


# def OV_Loss_old(
#     s,
#     noise=0.0,
#     sigma=0.0,
#     return_interior=False,
#     trace="true",
#     t_batches=1,
#     n_batches=1,
# ):

#     def s_sep(mu, t, x, params):
#         mu_t = jnp.concatenate([mu, t])
#         return s(mu_t, x, params)

#     s_dt = jacrev(s_sep, 1)
#     s_dt_Vx = vmap(s_dt, (None, None, 0, None))

#     s_Vx = vmap(s, in_axes=(None, 0, None))
#     s_dx = jacrev(s, 1)
#     s_dx_Vx = vmap(s_dx, in_axes=(None, 0, None))

#     if trace == "hutch":
#         trace_dds = hess_trace_estimator(s, argnum=1)
#         trace_dds_Vx = vmap(trace_dds, (None, None, 0, None))
#     else:
#         trace_dds = tracewrap(jacfwd(s_dx, 1))
#         trace_dds_Vx = vmap(trace_dds, (None, 0, None))
#         if n_batches > 1:
#             trace_dds_Vx = batchmap(trace_dds_Vx, n_batches, argnum=1)
#     if n_batches > 1:
#         s_dt_Vx = batchmap(s_dt_Vx, n_batches, argnum=2)
#         s_Vx = batchmap(s_Vx, n_batches, argnum=1)
#         s_dx_Vx = batchmap(s_dx_Vx, n_batches, argnum=1)

#     def loss_fn(params, x_t_batch, mu, t_batch, quad_weights, key):

#         T, N, D = x_t_batch.shape
#         T, MT = t_batch.shape
#         mu_t_0, mu_t_1 = jnp.concatenate([mu, t_batch[0]]), jnp.concatenate(
#             [mu, t_batch[-1]]
#         )

#         bound = s_Vx(mu_t_0, x_t_batch[0], params) - s_Vx(mu_t_1, x_t_batch[-1], params)

#         x_t_batch = x_t_batch[1:-1]
#         t_batch = t_batch[1:-1]

#         xt_tensor = rearrange(x_t_batch, "T N D -> T (N D)")
#         xt_tensor = jnp.hstack([xt_tensor, t_batch])

#         def interior_loss(xt_tensor):
#             x_batch, t = xt_tensor[:-MT], xt_tensor[-MT:]
#             x_batch = rearrange(x_batch, "(N D) -> N D", D=D)

#             mu_t = jnp.concatenate([mu, t])

#             ut = s_dt_Vx(mu, t, x_batch, params)
#             ut = jnp.squeeze(ut)

#             # entropic
#             if sigma > 0.0:
#                 if trace == "hutch":
#                     seed = jax.random.uniform(key, shape=()) + mu + t
#                     seed = jnp.linalg.norm(seed)
#                     keyt = jax.random.PRNGKey((seed * 1e8).astype(int))
#                     gu, tra = trace_dds_Vx(keyt, mu_t, x_batch, params)
#                 else:
#                     tra = trace_dds_Vx(mu_t, x_batch, params)
#                     gu = s_dx_Vx(mu_t, x_batch, params)

#                 ent = tra * sigma**2 * 0.5
#             else:
#                 gu = s_dx_Vx(mu_t, x_batch, params)
#                 ent = 0.0

#             gu = 0.5 * jnp.sum(gu**2, axis=1)

#             return (ut + gu + ent).mean()

#         interior_loss = vmap(interior_loss)
#         if t_batches > 1:
#             interior_loss = batchmap(interior_loss, t_batches)
#         interior = interior_loss(xt_tensor)

#         if quad_weights is not None:
#             loss_interior = (interior * quad_weights).sum()
#         else:
#             loss_interior = interior.mean()

#         loss = bound.mean() + loss_interior

#         if return_interior:
#             return loss, interior

#         return loss

#     return loss_fn
