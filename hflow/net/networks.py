from collections.abc import Iterable
from typing import Callable, List, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers

from hflow.net.layers import (CoLoRA, FiLM, Fourier_Random, Lipswish, Periodic,
                              Rational, Siren)


class DNN(nn.Module):
    width: int
    layers: List[str]
    out_dim: int
    activation: str = 'swish'
    period: Optional[jnp.ndarray] = None
    w0: float = 10.0
    rank: int = 1
    full: bool = False
    squeeze: bool = False
    bias: bool = True
    last_activation: Optional[str] = None
    w_init: str = 'lecun'

    @nn.compact
    def __call__(self, x):
        depth = len(self.layers)
        width = self.width
        w_init = get_init(self.w_init, self.w0)

        A = get_activation(self.activation)
        for i, layer in enumerate(self.layers):
            is_last = i == depth - 1
            if is_last:
                width = self.out_dim
            L = get_layer(layer=layer, width=width, w_init=w_init,
                          period=self.period, rank=self.rank, full=self.full, bias=self.bias, w0=self.w0)
            x = L(x)
            if not is_last:
                x = A(x)

        if self.last_activation is not None and self.last_activation != 'none':
            x = get_activation(self.last_activation)(x)

        if self.squeeze:
            x = jnp.squeeze(x)
        return x


def get_init(init, w0):

    if init is None or init == 'lecun':
        w = initializers.lecun_normal()
    elif init == 'ortho':
        w = initializers.orthogonal()
    elif init == 'normal':
        w = initializers.truncated_normal(w0)
    elif init == 'he':
        w = initializers.he_normal()
    return w


def get_layer(layer, width, w_init, period=None, rank=1, full=False, bias=True, w0=10.0):
    if layer == 'D':
        L = nn.Dense(width, use_bias=bias,  kernel_init=w_init)
    elif layer == 'P':
        L = Periodic(width, period=period, use_bias=bias, w_init=w_init)
    elif layer == 'C':
        L = CoLoRA(width, rank, full, use_bias=bias, w_init=w_init)
    elif layer == 'F':
        L = FiLM(width, full, use_bias=bias, w_init=w_init)
    elif layer == 'Fr':
        L = Fourier_Random(width, variance=w0)
    else:
        raise Exception(f"unknown layer: {layer}")
    return L


def get_activation(activation):
    if activation == 'relu':
        a = jax.nn.relu
    elif activation == 'tanh':
        a = jax.nn.tanh
    elif activation == 'sigmoid':
        a = jax.nn.sigmoid
    elif activation == 'elu':
        a = jax.nn.elu
    elif activation == 'selu':
        a = jax.nn.selu
    elif activation == 'rational':
        a = Rational()
    elif activation == 'swish':
        a = jax.nn.swish
    elif activation == 'siren':
        a = Siren(omega=10.0)
    elif activation == 'sin':
        a = jnp.sin
    elif activation == 'hswish':
        a = jax.nn.hard_swish
    elif activation == 'lipswish':
        return Lipswish()
    return a
