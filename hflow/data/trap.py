
import jax
import jax.numpy as jnp
from jax import vmap


def get_ic_trap(n_particles, key):
    var = 0.1
    ic = jax.random.normal(key, (n_particles, ))
    dd = jnp.arange(n_particles)
    mu = (9/10)+21/(10*(n_particles-1))*dd
    return (ic-mu)*var


def get_trap(n_particles, mu):

    alpha = -0.5
    D = 1e-2

    def a(t):
        return (5/4)*jnp.sin(jnp.pi*t+(3/2)+mu*jnp.cos(t*jnp.pi*2))

    def g_fn(t, x):
        return (a(t) - x)**3

    def kernel_fn(x1, x2):
        return alpha/n_particles*(x1-x2)

    kernel_fn = vmap(kernel_fn, (0, None))
    kernel_fn = vmap(kernel_fn, (None, 0))

    def drift(t, y, *args):
        t1 = g_fn(t, y)
        K = kernel_fn(y, y)
        t2 = jnp.sum(K, axis=0)
        return t1 + t2

    def diffusion(t, y, *args):

        return jnp.sqrt(2*D)*jnp.ones_like(y)

    return drift, diffusion


def get_ic_trap2(n_particles, key):
    var = 0.1
    ic = jax.random.normal(key, (n_particles, ))
    dd = jnp.arange(n_particles)
    mu = (9/10)+21/(10*(n_particles-1))*dd
    return (ic-mu)*var


def get_trap2(n_particles, mu):

    alpha = -0.5
    D = 1e-2

    def a(t):
        return (5/4)*jnp.sin(jnp.pi*t+(3/2))

    def g_fn(t, x):
        t = t * mu
        return (a(t) - x)**3

    def kernel_fn(x1, x2):
        return alpha/n_particles*(x1-x2)

    kernel_fn = vmap(kernel_fn, (0, None))
    kernel_fn = vmap(kernel_fn, (None, 0))

    def drift(t, y, *args):
        t1 = g_fn(t, y)
        K = kernel_fn(y, y)
        t2 = jnp.sum(K, axis=0)
        return t1 + t2

    def diffusion(t, y, *args):

        return jnp.sqrt(2*D)*jnp.ones_like(y)

    return drift, diffusion
