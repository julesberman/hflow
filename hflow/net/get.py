import jax.numpy as jnp
import numpy as np

from hflow.config import Config, Network
from hflow.net.build import build_colora


def get_network(cfg: Config, data, key):

    if cfg.unet.model == 'colora':
        u_fn, h_fn, params_init = get_colora(cfg,
                                             cfg.unet, cfg.hnet, data, key)

        def s_fn(t, x, params):
            psi, theta = params
            phi = h_fn(psi, t)
            return jnp.squeeze(u_fn(theta, phi, x))
    elif cfg.unet.model == 'mlp':
        u_fn, h_fn, params_init = get_colora(cfg,
                                             cfg.unet, cfg.hnet, data, key)

        def s_fn(t, x, params):
            psi, theta = params
            phi = h_fn(psi, t)
            return jnp.squeeze(u_fn(theta, phi, x))
    return s_fn, params_init


def get_colora(cfg: Config, unet: Network, hnet: Network, data, key):

    sols, mu, t = data
    MT = mu.shape[-1] + 1
    M, T, N, D = sols.shape

    if cfg.loss.loss_fn == 'ov':
        x_dim = D
        mu_t_dim = MT
        u_dim = 1
    elif cfg.loss.loss_fn == 'ncsm':
        x_dim = D
        mu_t_dim = MT+1  # one more for sigma
        u_dim = D

    period = np.asarray([1.0]*x_dim)

    u_config = {'width': unet.width,
                'layers': unet.layers,
                'activation': unet.activation,
                'last_activation': unet.activation,
                'w0': unet.w0,
                'bias': unet.bias,
                'period': period,
                'w_init': unet.w_init,
                }

    h_config = {'width': hnet.width,
                'layers': hnet.layers,
                'activation': hnet.activation,
                'last_activation': hnet.activation,
                'w0': hnet.w0,
                'bias': hnet.bias,
                'w_init': hnet.w_init, }

    u_fn, h_fn, theta_init, psi_init = build_colora(
        u_config, h_config, x_dim, mu_t_dim, u_dim, rank=unet.rank, key=key, full=unet.full)

    psi_theta_init = (psi_init, theta_init)

    return u_fn, h_fn, psi_theta_init


def get_mlp(cfg: Config, unet: Network, hnet: Network, data, key):

    sols, mu, t = data
    MT = mu.shape[-1] + 1
    M, T, N, D = sols.shape

    if cfg.loss.loss_fn == 'ov':
        x_dim = D
        mu_t_dim = MT
        u_dim = 1
    elif cfg.loss.loss_fn == 'ncsm':
        x_dim = D
        mu_t_dim = MT+1  # one more for sigma
        u_dim = D

    period = np.asarray([1.0]*x_dim)

    u_config = {'width': unet.width,
                'layers': unet.layers,
                'activation': unet.activation,
                'last_activation': unet.activation,
                'w0': unet.w0,
                'bias': unet.bias,
                'period': period,
                'w_init': unet.w_init,
                }

    h_config = {'width': hnet.width,
                'layers': hnet.layers,
                'activation': hnet.activation,
                'last_activation': hnet.activation,
                'w0': hnet.w0,
                'bias': hnet.bias,
                'w_init': hnet.w_init, }

    u_fn, h_fn, theta_init, psi_init = build_colora(
        u_config, h_config, x_dim, mu_t_dim, u_dim, rank=unet.rank, key=key, full=unet.full)

    psi_theta_init = (psi_init, theta_init)

    return u_fn, h_fn, psi_theta_init
