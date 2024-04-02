import jax.numpy as jnp
import numpy as np

from hflow.config import Config, Network
from hflow.net.build import build_colora


def get_network(cfg: Config, data, key):

    u_fn, h_fn, params_init = get_colora(
        cfg.unet, cfg.hnet, data, key)

    def s_fn(t, x, params):
        psi, theta = params
        phi = h_fn(psi, t)
        return jnp.squeeze(u_fn(theta, phi, x))

    return s_fn, params_init


def get_colora(unet: Network, hnet: Network, data, key):

    sols, mu, t = data
    MT = mu.shape[-1] + 1
    M, T, N, D = sols.shape

    x_dim = D
    mu_t_dim = MT
    u_dim = 1
    rank = unet.rank
    period = np.asarray([1.0]*x_dim)

    u_config = {'width': unet.width, 'layers': unet.layers}
    h_config = {'width': hnet.width, 'layers': hnet.layers}

    u_fn, h_fn, theta_init, psi_init = build_colora(
        u_config, h_config, x_dim, mu_t_dim, u_dim, 
        period=period, rank=rank, key=key, full=unet.full, bias=unet.bias)

    psi_theta_init = (psi_init, theta_init)

    return u_fn, h_fn, psi_theta_init
