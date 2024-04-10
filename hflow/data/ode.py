

from time import time

import jax
import jax.numpy as jnp
from jax import jit


def odeint_rk4(fn, y0, t, downsampler=lambda x: x):
    @jit
    def rk4(carry, t):
        y, t_prev = carry
        h = t - t_prev
        k1 = fn(t_prev, y)
        k2 = fn(t_prev + h / 2, y + h * k1 / 2)
        k3 = fn(t_prev + h / 2, y + h * k2 / 2)
        k4 = fn(t, y + h * k3)
        y = y + 1.0 / 6.0 * h * (k1 + 2 * k2 + 2 * k3 + k4)
        yd = downsampler(y)
        return (y, t), yd

    (yf, _), y = jax.lax.scan(rk4, (y0, jnp.array(t[0])), t)
    return y


def odeint_rk4_key(fn, y0, t, key, lambda_fn=lambda x, k: x):
    @jit
    def rk4(carry, t):
        y, t_prev, key = carry
        h = t - t_prev
        key, subkey = jax.random.split(key)

        k1 = fn(t_prev, y, subkey)
        k2 = fn(t_prev + h / 2, y + h * k1 / 2, subkey)
        k3 = fn(t_prev + h / 2, y + h * k2 / 2, subkey)
        k4 = fn(t, y + h * k3,  subkey)

        y = y + 1.0 / 6.0 * h * (k1 + 2 * k2 + 2 * k3 + k4)
        y = lambda_fn(y, subkey)
        return (y, t, key), y

    (yf, _, _), y = jax.lax.scan(rk4, (y0, jnp.array(t[0]), key), t[1:])
    y = jnp.insert(y, 0, y0, axis=0)  # prepend y0
    return y


def odeint_euler_key(fn, y0, t, key, lambda_fn=lambda x: x):
    @jit
    def rk4(carry, t):
        y, t_prev, key = carry
        h = t - t_prev
        key, subkey = jax.random.split(key)
        k1 = fn(t_prev, y, subkey)
        y = y + h * k1
        yd = lambda_fn(y)
        return (y, t, key), yd

    (yf, _, _), y = jax.lax.scan(rk4, (y0, jnp.array(t[0]), key), t)

    return y
