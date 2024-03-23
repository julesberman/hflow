
import random

import jax
import jax.numpy as jnp
from jax import vmap

from hflow.data.ode import odeint_rk4


def rand_f_old(t, x, K, sigma, t_seed=0):
    x = jnp.squeeze(x)
    t_seed = jnp.array(t*1e6+t_seed, int)
    t_key = jax.random.PRNGKey(t_seed)
    W_s, W_c = jax.random.normal(t_key, shape=(2,))
    f = jnp.zeros_like(x)
    for m in range(1, K):
        f += jnp.sin(m*x)*W_s + jnp.cos(m*x)*W_c
    return f*sigma


def gen_rand_f(period, K, sigma, key):
    def rand_f(x, y):
        px, py = period
        f = 0.0
        akey = key
        for m in range(1, K):
            for n in range(1, K):
                skey, akey = jax.random.split(akey)
                W_s, W_c = jax.random.normal(skey, shape=(2,))
                f += jnp.sin(m*x*jnp.pi/px)*jnp.sin(n*y*jnp.pi/py)*W_s + \
                    1j*jnp.cos(m*x*jnp.pi/px)*jnp.cos(n*y*jnp.pi/py)*W_c
        return f.real*sigma
    return rand_f


def solve_sburgers(key, X, ic, t_eval, period, sigma, modes, nu=5e-3):

    if key is None:
        key = jax.random.PRNGKey(random.randint(0, 1e6))

    rand_f = gen_rand_f(period, modes, sigma, key)
    rand_f = vmap(rand_f,  (0, None))

    dx = X[1]-X[0]
    N = len(X)

    def extend_periodic(u):
        u_ext = jnp.zeros(N+2)
        u_ext = u_ext.at[1:-1].set(u)
        u_ext = u_ext.at[0].set(u[-1])
        u_ext = u_ext.at[-1].set(u[0])

        return u_ext

    def burgers_rhs(t, u):
        f = jnp.squeeze(rand_f(X, t))
        u = extend_periodic(u)
        u_x = (u[2:] - u[0:-2]) / (2*dx)
        u_xx = (u[2:] - 2*u[1:-1] + u[0:-2]) / (dx**2)
        u_t = -u[1:-1]*u_x + nu*u_xx + f
        return u_t

    y = odeint_rk4(burgers_rhs, ic, t_eval)

    return y
