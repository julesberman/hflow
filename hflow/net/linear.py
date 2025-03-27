import jax
import jax.numpy as jnp
from flax import linen as nn
from hflow.net.dnn import MLP
from typing import Callable, List, Optional


class LinearFourier(nn.Module):
    """Parameterizes a 2D function using a linear Fourier basis.

    f(x, y) = bias
              + sum_{i=1 to width} sum_{j=1 to width} [
                  cos_coeff[i, j] * cos(i*x + j*y)
                + sin_coeff[i, j] * sin(i*x + j*y)
              ]

    Attributes:
        width: number of Fourier modes along each axis.
        use_bias: whether to include a single learnable bias term.
    """
    width: int
    use_bias: bool 
    cond_features:  List[int]


    @nn.compact
    def __call__(self, x, cond_x):
  
        coords = x
        width = self.width // 2



        Hyper_Net_Head = MLP(
            features=self.cond_features,
            activate_last=False,
            out_features= width *  width *2,
            activation='swish',
            use_bias=True,
        )

        s_b = Hyper_Net_Head(cond_x).reshape(2, width, width)
        cos_coeff, sin_coeff = s_b[0], s_b[1]
    

        # # Learnable cosine and sine coefficients for each (i, j).
        # # Shape: (width, width).
        # cos_coeff = self.param(
        #     'cos_coeff',
        #     lambda rng, shape: jnp.zeros(shape),
        #     (width, width)
        # )
        # sin_coeff = self.param(
        #     'sin_coeff',
        #     lambda rng, shape: jnp.zeros(shape),
        #     (width, width)
        # )

        # Optional bias (single scalar).
        if self.use_bias:
            bias = self.param(
                'bias',
                lambda rng, shape: jnp.zeros(shape),
                (1,)
            )
        else:
            bias = jnp.zeros((1,))

        x, y = coords[0], coords[1]

        # Frequencies: 1, 2, ..., width
        i_vals = jnp.arange(1, width + 1, dtype=coords.dtype)  # shape (width,)
        j_vals = jnp.arange(1, width + 1, dtype=coords.dtype)  # shape (width,)

        # Compute phases for all i, j:
        # i_vals in one dimension, j_vals in the other -> shape (width, width).
        # phases[w1, w2] = i_vals[w1]*x + j_vals[w2]*y
        phases = i_vals[:, None] * x + j_vals[None, :] * y  # shape: (width, width)

        # cos_coeff[i, j] * cos(phases[i, j]) + sin_coeff[i, j] * sin(phases[i, j])
        # then sum over i and j.
        total_cos = cos_coeff * jnp.cos(phases)
        total_sin = sin_coeff * jnp.sin(phases)
        fourier_sum = jnp.sum(total_cos + total_sin)

        return fourier_sum + bias[0]