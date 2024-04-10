import random

import flax
import jax
import jax.numpy as jnp
from jax import vmap
from jax.flatten_util import ravel_pytree

from hflow.misc.partition import merge, partition


def init_net(net, input_dim, key=None):
    if key is None:
        key = jax.random.PRNGKey(random.randint(0, 10_000))
    pt = jnp.zeros(input_dim)
    theta_init = net.init(key, pt)
    f = net.apply
    return theta_init, f


def gen_n_inits(N, input_dim, net, key=None):
    if key is None:
        key = jax.random.PRNGKey(random.randint(0, 10_000))

    pt = jnp.zeros(input_dim)
    keys = jax.random.split(key, num=N)
    init_first = net.init(key, pt)
    _, unravel = ravel_pytree(init_first)

    def gen_init(key):
        init = net.init(key, pt)
        return ravel_pytree(init)[0]

    inits = vmap(gen_init)(keys)
    f_inits = [unravel(i) for i in inits]

    return f_inits


def split(theta_phi, filter_list):

    if not isinstance(filter_list, list):
        filter_list = [filter_list]

    def filter_rn(m, leaf_key, p):
        return leaf_key in filter_list

    _, theta_phi = flax.core.pop(theta_phi, 'params')
    phi, theta = partition(filter_rn, theta_phi)

    phi = {'params': phi}
    theta = {'params': theta}
    return phi, theta


def merge(phi, theta):
    theta_phi = merge(phi['params'], theta['params'])
    theta_phi = {'params': theta_phi}
    return theta_phi
