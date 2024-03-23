import numpy as np

from hflow.config import Network
from hflow.net.build import build_colora


def get_colora(unet: Network, hnet: Network, dataset, key):

    mu_t, x = dataset.sample()
    B, MT = mu_t.shape
    B, N, Q, D = x.shape

    x_dim = D
    mu_t_dim = MT
    u_dim = Q
    rank = unet.rank
    period = np.asarray([1.0]*x_dim)

    u_config = {'width': unet.width, 'layers': unet.layers}
    h_config = {'width': hnet.width, 'layers': hnet.layers}

    u_fn, h_fn, theta_init, psi_init = build_colora(
        u_config, h_config, x_dim, mu_t_dim, u_dim, period=period, rank=rank, key=key)

    psi_theta_init = (psi_init, theta_init)

    return u_fn, h_fn, psi_theta_init
