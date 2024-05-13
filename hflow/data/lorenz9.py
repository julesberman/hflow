import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.experimental.host_callback import id_print


def get_ic_lorenz9d(key):
    n_particles = 9
    var = 2e-2
    ic = jax.random.normal(key, (n_particles, ))*var
    return ic


def get_lorenz9d(mu):

    def drift(t, y, *args):
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = y

        a = 0.5
        s = 0.5
        b1 = 4*(1+a**2)/(1+2*a**2)
        b2 = (1+2*a**2)/(2*(1+a**2))
        b3 = 2*((1-a**2)/(1+a**2))
        b4 = a**2/(1+a**2)
        b5 = (8*a**2)/(1+2*a**2)
        b6 = 4/(1+2*a**2)

        r = mu

        c1_dot = -s*b1*c1 - c2*c4 + b4*c4**2 + b3*c3*c5 - s*b2*c7
        c2_dot = -s*c2 + c1*c4 - c2*c5 + c4*c5 - s*c9/2
        c3_dot = -s*b1*c3 + c2*c4 - b4*c2**2 - b3*c1*c5 + s*b2*c8
        c4_dot = -s*c4 - c2*c3 - c2*c5 + c4*c5 + s*c9/2
        c5_dot = -s*b5*c5 + c2**2/2 - c4**2/2
        c6_dot = -b6*c6 + c2*c9 - c4*c9
        c7_dot = -b1*c7 - r*c1 + 2*c5*c8 - c4*c9
        c8_dot = -b1*c8 + r*c3 - 2*c5*c7 + c2*c9
        c9_dot = -c9 - r*c2 + r*c4 - 2*c2*c6 + 2*c4*c6 + c4*c7 - c2*c8
        return jnp.asarray([c1_dot, c2_dot, c3_dot, c4_dot, c5_dot, c6_dot, c7_dot, c8_dot, c9_dot])

    def diffusion(t, y, *args):

        return jnp.ones_like(y)*2e-2

    return drift, diffusion
