
from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import grad, jacfwd, jacrev, jit, jvp, vmap
from jax.experimental.host_callback import id_print

from hflow.config import Data, Loss
from hflow.misc.jax import batchmap, hess_trace_estimator, tracewrap


def FD_Loss(s_comb, bs_t, bs_n):

    def s(params, x, t, mu):
        mu_t = jnp.concatenate([mu, t])

        return s_comb(mu_t, x, params) - s_comb(mu_t, jnp.zeros_like(x), params)

    def nabla_s(params, x, t, mu):
        return jax.grad(lambda x: s(params, x, t, mu))(x)

    def nabla_s_squared(params, x, t, mu):
        return jnp.sum(nabla_s(params, x, t, mu)**2)

    def loss_fn(params, x_data, i_mu, t_data, mu_data, key):

        def expected_value(f, it1, it2, i_mu, key):

            # get a random subset of x at time t1
            t2 = t_data[it2]
            x = get_random_subset(key, x_data[:, it1, i_mu], bs_n)
            mu = mu_data[i_mu]
            # evaluate f at the time t2 at all x and average
            # signature of f should be (x, t, mu)
            return jnp.mean(jax.vmap(lambda _x: f(_x, t2, mu))(x))

        # def expected_value(f, t1, t2, i_mu, key):
        #     # get a random subset of x at time t1
        #     dt = t_data[1]
        #     it1 = jnp.int32(t1/dt)

        #     x = get_random_subset(key, x_data[:, it1, i_mu, :], bs_n)

        #     # evaluate f at the time t2 at all x and average
        #     # signature of f should be (x, t, mu)
        #     return jnp.mean(jax.vmap(lambda _x: f(_x, t2, mu))(x))

        def get_uniform_subset(arr, bs):
            N = len(arr)
            indices = jnp.int32(jnp.linspace(0, N-1, bs))
            subset = arr[indices]
            return subset

        def get_random_subset(k, arr, bs):
            N = len(arr)
            indices = jax.random.choice(k, N, shape=(bs,), replace=False)
            subset = arr[indices]
            return subset

        key, x_key, t_key = jax.random.split(key, 3)
        it_s = jnp.int32(get_uniform_subset(jnp.arange(len(t_data)), bs_t))
        # it_s = jnp.int32(get_random_subset(t_key, jnp.arange(nt), bs_t))
        # it_s = it_s.at[0].set(0)
        # it_s = it_s.at[-1].set(len(it_s)-1)
        # it_s = jnp.sort(it_s)
        times = t_data[it_s]
        mu = mu_data[i_mu]

        x_keys = jax.random.split(x_key, bs_t)     # uncorrelated samples
        # x_keys = jnp.array([x_key] * bs_t)  # trajectories

        def _s(x, t, mu): return s(params, x, t, mu)
        def E_s(t1, t2, key): return expected_value(_s, t1, t2, i_mu, key)
        E_s_v = jax.vmap(E_s)

        sum_En_snplus1 = jnp.sum(E_s_v(it_s[:-1], it_s[1:],  x_keys[:-1]))
        sum_Enplus1_sn = jnp.sum(E_s_v(it_s[1:],  it_s[:-1], x_keys[1:]))
        sum_En_sn = jnp.sum(E_s_v(it_s[:-1], it_s[:-1], x_keys[:-1]))
        sum_Enplus1_snplus1 = jnp.sum(E_s_v(it_s[1:],  it_s[1:],  x_keys[1:]))

        integration_weights = 0.5 * jnp.concatenate([jnp.array([times[1] - times[0]]),
                                                    (times[2:] - times[:-2]),
                                                    jnp.array([times[-1] - times[-2]])])

        def _grad_s_sq(x, t, mu): return nabla_s_squared(params, x, t, mu)

        def E_grad_s_sq(t1, t2, key): return expected_value(
            _grad_s_sq, t1, t2, i_mu, key)

        sum_En_gradsn = 0.5 * \
            jnp.sum(integration_weights *
                    jax.vmap(E_grad_s_sq)(it_s, it_s, x_keys))

        loss = 0
        loss += (+ 0.5 * sum_En_snplus1
                 - 0.5 * sum_Enplus1_sn
                 + 0.5 * sum_En_sn
                 - 0.5 * sum_Enplus1_snplus1)
        loss += sum_En_gradsn

        return loss

    return loss_fn
