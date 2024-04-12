
import random
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap

from hflow.data.ode import odeint_euler_key


def get_f_besov_x(key, x_space, modes, s=1.0, kappa=1.0, p=2):
    d = 1
    pi = jnp.pi
    L = x_space[-1] - x_space[0]
    u0 = 0.0
    xi_k = jax.random.normal(key, shape=(2*modes,))

    def sin(k):
        return jnp.sin(2*pi*k*x_space/L) * jnp.sqrt(2/L)

    def cos(k):
        return jnp.cos(2*pi*k*x_space/L) * jnp.sqrt(2/L)
    for k in range(1, modes):
        u0 += (k**(-s/d-1/p+1/2) * kappa**(-1/p)) * \
            (xi_k[k-1] * cos(k) + xi_k[k-1+modes] * sin(k))
    return u0


def solve_sburgers(N, sub_N, t_eval, modes, sigma, nu, key):
    X = jnp.linspace(0.0, 1.0, N)

    def ic_fn(x):
        return jnp.squeeze(jnp.exp(-20*(x-0.5)**2))

    ic = ic_fn(X)
    dx = X[1]-X[0]
    N = len(X)
    dt = t_eval[1]

    def extend_periodic(u):
        u_ext = jnp.zeros(N+2)
        u_ext = u_ext.at[1:-1].set(u)
        u_ext = u_ext.at[0].set(u[-1])
        u_ext = u_ext.at[-1].set(u[0])

        return u_ext

    def burgers_rhs(t, u, key):
        u = extend_periodic(u)
        f = get_f_besov_x(key, X, modes)*sigma*(1/jnp.sqrt(dt))
        u_x = ((u[1:]-u[0:-1])/dx)[:-1]
        u_xx = (u[2:] - 2*u[1:-1] + u[0:-2]) / (dx**2)
        u_t = -u[1:-1]*u_x + nu*u_xx + f
        return u_t

    def lambda_fn(y):
        return y[::sub_N]

    y = odeint_euler_key(burgers_rhs, ic, t_eval, key, lambda_fn=lambda_fn)

    return y


def solve_sburgers_samples(n_samples, mus, N, sub_N, sigma, modes, t_eval, key):

    sols = []
    for mu in mus:
        keys = jax.random.split(key, num=n_samples)
        jit_solver = vmap(
            jit(partial(solve_sburgers, N, sub_N, t_eval, modes, sigma, mu)))
        s = jit_solver(keys)
        sols.append(s)

    return jnp.asarray(sols)
