import jax.numpy as jnp
import numpy as np

from hflow.config import Config, Network
from hflow.net.dnn import DNN
from hflow.net.linear import LinearFourier
from hflow.net.unet import UNet
import hflow.io.result as R
from hflow.config import Optimizer
import jax
from hflow.net.resnet import ResNeSt50FastSmall, ResNet18

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


    if net.arch == 'mlp':
        net = DNN(
            features=[net.width] * net.depth,
            activation=net.activation,
            cond_features=net.cond_features,
            use_bias=net.bias,
            cond_in=net.cond_in,
            out_features=out_dim,
        )
    elif net.arch == 'linear':
        net = LinearFourier(
            width=net.width,
            use_bias=net.bias,
            cond_features=net.cond_features,
        )

    elif net.arch == 'unet':
        net = UNet(
            # feature_depths=[128, 256, 512],
            feature_depths=[32, 64],
            emb_features=256,
            num_res_blocks=2,
            num_middle_res_blocks=2,
            out_channels=1,

        )

    elif net.arch == 'resnet':
        net = ResNeSt50FastSmall(n_classes=1, hidden_sizes=[16, 32, 32])

    x_in = jnp.zeros(x_dim)
    cond_x = jnp.zeros(mu_t_dim)

    print('x_in, x_cond', x_in.shape, cond_x.shape)

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
