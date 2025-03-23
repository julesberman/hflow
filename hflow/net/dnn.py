from collections.abc import Iterable
from typing import Callable, List, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

from hflow.net.networks import get_activation, get_init


class DNN(nn.Module):
    features: List[int]
    cond_features:  List[int]
    activation: str = "swish"
    use_bias: bool = True
    kernel_init: str = "lecun"
    activate_last: bool = False
    residual: bool = False
    out_features: int | None = None
    cond_in: str = "append"

    @nn.compact
    def __call__(self, x, cond_x):


        if self.cond_in == "append":
            x = jnp.concatenate([x, cond_x])
            cond_x = None


        A = get_activation(self.activation)
        kernel_init = get_init(self.kernel_init)
        last_x = None
        for i, feats in enumerate(self.features):
      
            D = nn.Dense(
                feats,
                use_bias=self.use_bias,
                kernel_init=kernel_init,
            )
            last_x = x
            x = D(x)
            x = A(x)
            # if self.norm_layer is not None:
            #     N = get_norm_layer(self.norm_layer)
            #     x = N(x)

            if self.cond_in != "append" and cond_x is not None:
                Hyper_Net_Head = MLP(
                    features=self.cond_features,
                    activate_last=False,
                    out_features=feats * 2,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    kernel_init=self.kernel_init,
                )

                s_b = Hyper_Net_Head(cond_x).reshape(-1, 2, feats)
                scale, bias = s_b[:, 0], s_b[:, 1]
                scale = scale.reshape(-1, feats)
                bias = bias.reshape(-1, feats)
                x = x * scale + bias

            if self.residual and last_x is not None:
                if x.shape == last_x.shape:
                    x = x + last_x

        D = nn.Dense(
            self.out_features,
            use_bias=self.use_bias,
            kernel_init=kernel_init,
        )

        x = D(x)

        return x


class MLP(nn.Module):
    features: List[int]
    out_features: int
    activation: str = "swish"
    use_bias: bool = True
    kernel_init: str = "lecun"
    activate_last: bool = False

    @nn.compact
    def __call__(self, x):
        kernel_init = get_init(self.kernel_init)
        A = get_activation(self.activation)

        for i, feats in enumerate(self.features):
            D = nn.Dense(
                feats,
                use_bias=self.use_bias,
                kernel_init=kernel_init,
            )
            x = D(x)
            x = A(x)

        D = nn.Dense(
            self.out_features,
            use_bias=self.use_bias,
            kernel_init=kernel_init,
        )
        x = D(x)

        if self.activate_last:
            x = A(x)

        return x
