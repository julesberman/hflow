
import jax
import jax.numpy as jnp


def get_ic_bi(key):
    mu_0 = jnp.asarray([0, 10])
    ic = jax.random.normal(key, (2,))
    ic = (ic*0.5) - mu_0
    return ic


def get_2d_bi(mu):
    def drift(t, y, *args):
        xi, gamma, w = 0.20, mu, 1.0
        x1, x2 = y
        x1_dot = x2
        x2_dot = -2*xi*w*x2 + w**2*x1 - w**2*gamma*x1**3
        return jnp.asarray([x1_dot, x2_dot])

    def diffusion(t, y, *args):
        return jnp.asarray([0, 1])

    return drift, diffusion


def get_ic_lin(key):
    mu_0 = jnp.asarray([1, 1, 0, 0])
    ic = jax.random.normal(key, (4,))
    ic = (ic*0.1) - mu_0
    return ic


def get_2d_lin(mu):
    def drift(t, y, *args):
        x1, x2, x3, x4 = y
        x1_dot = x3
        x2_dot = x4
        x3_dot = -mu**2*x1
        x4_dot = -mu**2*x2
        return jnp.asarray([x1_dot, x2_dot, x3_dot, x4_dot])

    def diffusion(t, y, *args):
        return jnp.asarray([0, 0, 1, 1])*5e-2

    return drift, diffusion
