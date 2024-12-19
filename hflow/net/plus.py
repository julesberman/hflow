from collections.abc import Iterable
from typing import Callable, List, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

from hflow.net.networks import get_activation, get_norm_layer


class PLUS(nn.Module):
    features: List[int]
    out_features: int
    h_features: List[int] | None = None
    activation: str = "swish"
    use_bias: bool = True
    activate_last: bool = False
    norm_layer: str | None = None
    residual: bool = False
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, conditional_x, train=False):

        A = get_activation(self.activation)

        x_first = x
        last_x = None

        for i, feats in enumerate(self.features):
            D = nn.Dense(
                feats,
                use_bias=self.use_bias,
            )
            last_x = x
            x = D(x)
            x = A(x)
            if self.norm_layer is not None:
                N = get_norm_layer(self.norm_layer)
                x = N(x)

            if self.h_features is not None:
                Hyper_Net_Head = MLP(
                    features=self.h_features,
                    out_features=feats * 2,
                    activation=self.activation,
                    use_bias=self.use_bias,
                )
                scale, bias = Hyper_Net_Head(conditional_x).reshape(2, feats)
                x = x * scale + bias

            if self.residual and last_x is not None:
                if x.shape == last_x.shape:
                    x = x + last_x

        D = nn.Dense(
            self.out_features,
            use_bias=self.use_bias,
        )
        x = D(x)

        if self.activate_last:
            x = A(x)

        return x


class MLP(nn.Module):
    features: List[int]
    out_features: int
    activation: str = "swish"
    use_bias: bool = True
    kernel_init: str = "lecun"
    bias_init: str = "zero"
    activate_last: bool = False

    @nn.compact
    def __call__(self, x):

        A = get_activation(self.activation)

        for i, feats in enumerate(self.features):
            D = nn.Dense(
                feats,
                use_bias=self.use_bias,
            )
            x = D(x)
            x = A(x)

        D = nn.Dense(
            self.out_features,
            use_bias=self.use_bias,
        )
        x = D(x)

        if self.activate_last:
            x = A(x)

        return x
