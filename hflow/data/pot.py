import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
from einops import rearrange


def rk4(x, f, t, dt, eps, key):
    # signature of f is f(x, t)
    k1 = f(x, t)
    k2 = f(x + dt/2 * k1, t + dt/2)
    k3 = f(x + dt/2 * k2, t + dt/2)
    k4 = f(x + dt * k3, t + dt)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def euler(x, f, t, dt, eps, key):
    return x + dt * f(x, t)


def euler_marujama(x, f, t, dt, eps, key):
    return x + dt * f(x, t) + eps * jnp.sqrt(dt) * jax.random.normal(key, x.shape)


def generate_sample_s(x0, v, times, L, key, eps):
    def step(carry, t_next):
        x, t_prev, key = carry
        dt = t_next - t_prev
        # x_new = rk4(x, v, t_prev, dt, eps, key)
        x_new = euler_marujama(x, v, t_prev, dt, eps, key)
        if L != 0:
            x_new = jnp.mod(x_new, L)
        new_carry = (x_new, t_next, jax.random.split(key)[0])
        return new_carry, x_new

    init = (x0, times[0], key)
    carry, xs = jax.lax.scan(step, init, times[1:])
    xs = jnp.vstack([x0[None, ...], xs])
    return xs


def styblinski_tang(x):
    return 0.2 * 0.5 * jnp.sum(x**4 - 16 * x**2 + 5 * x, axis=-1)


def oakley_ohagan(x):
    return 0.2 * 5 * jnp.sum(jnp.sin(x) + jnp.cos(x) + x**2 + x, axis=-1)


def combined_potential(x, t):
    return (jnp.sin(jnp.pi * t/2)**2 * styblinski_tang(x)
            + jnp.cos(jnp.pi * t/2)**2 * oakley_ohagan(x)).sum()


def analytic_potential(n_samples, times, key, d, var):
    x0 = jax.random.normal(key, (n_samples, 1, d)) * jnp.sqrt(var)
    def grad_s(x, t): return - jax.grad(combined_potential)(x, t)
    return jax.vmap(lambda x, key: generate_sample_s(x, grad_s, times, 0, key, 0.0))(x0, jax.random.split(key, n_samples))
