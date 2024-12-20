from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers


class Periodic(nn.Module):

    width: int
    period: Optional[jnp.ndarray]
    param_dtype = jnp.float32
    use_bias: bool = True
    w_init: Callable = initializers.lecun_normal()

    @nn.compact
    def __call__(self, x):
        dim, f = x.shape[-1], self.width
        w_init = self.w_init
        if self.period is None:
            period = self.param(
                "period",
                w_init,
                (
                    f,
                    dim,
                ),
                self.param_dtype,
            )
        else:
            period = jnp.asarray(self.period)

        a = self.param("a", w_init, (f, dim), self.param_dtype)
        phi = self.param("c", w_init, (f, dim), self.param_dtype)

        omeg = jnp.pi * 2 / period
        o = a * jnp.cos(omeg * x + phi)
        if self.use_bias:
            b = self.param("b", w_init, (f, dim), self.param_dtype)
            o += b

        o = jnp.mean(o, axis=1)

        return o

        return out


class CoLoRA(nn.Module):

    width: int
    rank: int
    full: bool
    w_init: Callable = initializers.lecun_normal()
    b_init: Callable = initializers.zeros_init()
    use_bias: bool = True
    param_dtype = jnp.float32

    @nn.compact
    def __call__(self, X):
        D, K, r = X.shape[-1], self.width, self.rank

        w_init = self.w_init
        b_init = self.b_init
        z_init = initializers.zeros_init()

        W = self.param("W", w_init, (D, K), self.param_dtype)
        A = self.param("A", w_init, (D, r), self.param_dtype)
        B = self.param("B", w_init, (r, K), self.param_dtype)

        if self.full:
            n_alpha = self.rank
        else:
            n_alpha = 1

        alpha = self.param("alpha", z_init, (n_alpha,), self.param_dtype)

        AB = (A * alpha) @ B
        AB = AB  # / r
        W = W + AB

        out = X @ W

        if self.use_bias:
            b = self.param("b", b_init, (K,))
            b = jnp.broadcast_to(b, out.shape)
            out += b

        return out


class FiLM(nn.Module):

    width: int
    full: bool
    w_init: Callable = initializers.lecun_normal()
    b_init: Callable = initializers.zeros_init()
    use_bias: bool = True
    param_dtype = jnp.float32

    @nn.compact
    def __call__(self, X):
        D, K = X.shape[-1], self.width

        w_init = self.w_init
        b_init = self.b_init
        z_init = initializers.zeros_init()

        W = self.param("W", w_init, (D, K), self.param_dtype)

        if self.full:
            n_alpha = self.width
        else:
            n_alpha = 1

        alpha = self.param("alpha", z_init, (n_alpha * 2,), self.param_dtype)

        out = X @ W

        if self.use_bias:
            b = self.param("b", b_init, (K,))
            b = jnp.broadcast_to(b, out.shape)
            out += b

        gamma, beta = alpha[:n_alpha], alpha[n_alpha:]
        gamma += 1

        out = out * gamma + beta

        return out


class Rational(nn.Module):
    """
    Rational activation function
    ref: Nicolas Boullé, Yuji Nakatsukasa, and Alex Townsend,
        Rational neural networks,
        arXiv preprint arXiv:2004.01902 (2020).

    Source: https://github.com/yonesuke/RationalNets/blob/main/src/rationalnets/rational.py

    """

    p = 3

    @nn.compact
    def __call__(self, x):
        alpha_init = lambda *args: jnp.array(
            [1.1915, 1.5957, 0.5, 0.0218][: self.p + 1]
        )
        beta_init = lambda *args: jnp.array([2.383, 0.0, 1.0][: self.p])
        alpha = self.param("alpha", init_fn=alpha_init)
        beta = self.param("beta", init_fn=beta_init)

        return jnp.polyval(alpha, x) / jnp.polyval(beta, x)


class Siren(nn.Module):
    omega: float = 1.0

    @nn.compact
    def __call__(self, x):

        return jnp.sin(self.omega * x)


class Lipswish(nn.Module):

    @nn.compact
    def __call__(self, x):
        return 0.909 * jax.nn.silu(x)


class Fourier_Random(nn.Module):

    features: int
    param_dtype = jnp.float32
    variance: int = None

    @nn.compact
    def __call__(self, x):
        f, dim = self.features, x.shape[-1]

        key = jax.random.PRNGKey(1)
        R = jax.random.truncated_normal(key, upper=10, lower=-10, shape=(f, dim))
        R = R * self.variance
        fs = f // 2

        s = jnp.sin(R[:fs] @ x)
        c = jnp.cos(R[fs:] @ x)
        return jnp.concatenate([s, c])
