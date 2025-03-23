import jax.numpy as jnp
import numpy as np

from hflow.config import Config, Network
from hflow.net.dnn import DNN
import hflow.io.result as R
from hflow.config import Optimizer
import jax
def get_network(cfg: Config, data, key):

    net = cfg.net

    sols, mu, t = data
    MT = mu.shape[-1] + 1
    M, T, N, D = sols.shape

    if "ov" in cfg.loss.loss_fn or cfg.loss.loss_fn == "dice":
        x_dim = D
        mu_t_dim = MT
        out_dim = 1
        if cfg.net.homo:
            out_dim = D
    elif (
        cfg.loss.loss_fn == "ncsm"
        or cfg.loss.loss_fn == "cfm"
        or cfg.loss.loss_fn == "si"
    ):
        x_dim = D
        mu_t_dim = MT + 1  # one more for sigma
        out_dim = D


    net = DNN(
        features=[net.width] * net.depth,
        activation=net.activation,
        cond_features=net.cond_features,
        use_bias=net.bias,
        cond_in=net.cond_in,
        out_features=out_dim,
    )

    x_in = jnp.zeros(x_dim)
    cond_x = jnp.zeros(mu_t_dim)

    params_init = net.init(key, x_in, cond_x)

    param_count = sum(x.size for x in jax.tree_leaves(params_init))
    print(f"n_params {param_count:,}")
    R.RESULT["n_params"] = param_count


    def s_fn(t, x, params):
        return jnp.squeeze(net.apply(params, x, t))

    final_net = s_fn

    if cfg.net.fix_u:
        def s_fn_fixed(t, x, params):
            return s_fn(t, x, params) - s_fn(t, x * 0 + 0.5, params)
        final_net = s_fn_fixed
        
    if cfg.net.homo:
        def s_fn_homo(t, x, params):
            f_x = s_fn(t, x, params)
            y = 0.5 * jnp.dot(x, f_x)
            return jnp.squeeze(y)

        final_net = s_fn_homo

    return final_net, params_init
