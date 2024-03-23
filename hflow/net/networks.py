from collections.abc import Iterable
from typing import Callable, List, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from hflow.net.layers import CoLoRA, Periodic


class DNN(nn.Module):
    width: int
    layers: List[str]
    out_dim: int
    activation: Callable = jax.nn.swish
    period: Optional[jnp.ndarray] = None
    rank: int = 1
    full: bool = False
    squeeze: bool = False
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        depth = len(self.layers)
        width = self.width

        A = self.activation
        for i, layer in enumerate(self.layers):
            is_last = i == depth - 1

            if isinstance(self.activation, Iterable):
                A = self.activation[i]

            if is_last:
                width = self.out_dim
            L = get_layer(layer=layer, width=width,
                          period=self.period, rank=self.rank, full=self.full, bias=self.bias)
            x = L(x)
            if not is_last:
                x = A(x)

        if self.squeeze:
            x = jnp.squeeze(x)
        return x


def get_layer(layer, width, period=None, rank=1, full=False, bias=True):
    if layer == 'D':
        L = nn.Dense(width, use_bias=bias)
    elif layer == 'P':
        L = Periodic(width, period=period, use_bias=bias)
    elif layer == 'C':
        L = CoLoRA(width, rank, full, use_bias=bias)
    else:
        raise Exception(f"unknown layer: {layer}")
    return L
