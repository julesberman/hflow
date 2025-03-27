import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from typing import Dict, Callable, Sequence, Any, Union, Optional
from einops import rearrange
from functools import partial
from dataclasses import dataclass, field


"""
Some Code ported from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_flax.py
"""


# Kernel initializer to use
def kernel_init(scale=1.0, dtype=jnp.float32):
    scale = max(scale, 1e-10)
    return nn.initializers.variance_scaling(
        scale=scale, mode="fan_avg", distribution="truncated_normal", dtype=dtype
    )


class FourierEmbedding(nn.Module):
    features: int
    scale: int = 16

    def setup(self):
        self.freqs = (
            jax.random.normal(
                jax.random.PRNGKey(42), (self.features // 2,), dtype=jnp.float32
            )
            * self.scale
        )

    def __call__(self, x):
        x = jax.lax.convert_element_type(x, jnp.float32)
        emb = x[:, None] * (2 * jnp.pi * self.freqs)[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class TimeProjection(nn.Module):
    features: int
    activation: Callable = jax.nn.gelu
    kernel_init: Callable = kernel_init(1.0)

    @nn.compact
    def __call__(self, x):
        x = nn.DenseGeneral(self.features, kernel_init=self.kernel_init)(x)
        x = self.activation(x)
        x = nn.DenseGeneral(self.features, kernel_init=self.kernel_init)(x)
        x = self.activation(x)
        return x





class ConvLayer(nn.Module):
    conv_type: str
    features: int
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)
    kernel_init: Callable = kernel_init(1.0)
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):

        self.conv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            precision=self.precision,
        )


    def __call__(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    features: int
    scale: int
    activation: Callable = jax.nn.swish
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    kernel_init: Callable = kernel_init(1.0)

    @nn.compact
    def __call__(self, x, residual=None):
        out = x

        B, H, W, C = x.shape
        out = jax.image.resize(
            x, (B, H * self.scale, W * self.scale, C), method="nearest"
        )
        out = ConvLayer(
            "conv",
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.dtype,
            precision=self.precision,
            kernel_init=self.kernel_init,
        )(out)
        if residual is not None:
            out = jnp.concatenate([out, residual], axis=-1)
        return out


class Downsample(nn.Module):
    features: int
    scale: int
    activation: Callable = jax.nn.swish
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    kernel_init: Callable = kernel_init(1.0)

    @nn.compact
    def __call__(self, x, residual=None):
        out = ConvLayer(
            "conv",
            features=self.features,
            kernel_size=(3, 3),
            strides=(2, 2),
            dtype=self.dtype,
            precision=self.precision,
            kernel_init=self.kernel_init,
        )(x)
        if residual is not None:
            if residual.shape[1] > out.shape[1]:
                residual = nn.avg_pool(
                    residual, window_shape=(2, 2), strides=(2, 2), padding="SAME"
                )
            out = jnp.concatenate([out, residual], axis=-1)
        return out


class ResidualBlock(nn.Module):
    conv_type: str
    features: int
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)
    padding: str = "SAME"
    activation: Callable = jax.nn.swish
    direction: str = None
    res: int = 2
    norm_groups: int = 8
    kernel_init: Callable = kernel_init(1.0)
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        if self.norm_groups > 0:
            norm = partial(nn.GroupNorm, self.norm_groups)
            self.norm1 = norm()
            self.norm2 = norm()
        else:
            norm = partial(nn.RMSNorm, 1e-5)
            self.norm1 = norm()
            self.norm2 = norm()

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        temb: jax.Array,
        textemb: jax.Array = None,
        extra_features: jax.Array = None,
    ):
        residual = x
        out = self.norm1(x)
        # out = nn.RMSNorm()(x)
        out = self.activation(out)

        out = ConvLayer(
            self.conv_type,
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            kernel_init=self.kernel_init,
            name="conv1",
            dtype=self.dtype,
            precision=self.precision,
        )(out)

        if temb is not None:
            temb = nn.DenseGeneral(
                features=self.features,
                name="temb_projection",
                dtype=self.dtype,
                precision=self.precision,
            )(temb)
            temb = jnp.expand_dims(jnp.expand_dims(temb, 1), 1)
            # scale, shift = jnp.split(temb, 2, axis=-1)
            # out = out * (1 + scale) + shift
            out = out + temb

        out = self.norm2(out)
        # out = nn.RMSNorm()(out)
        out = self.activation(out)

        out = ConvLayer(
            self.conv_type,
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            kernel_init=self.kernel_init,
            name="conv2",
            dtype=self.dtype,
            precision=self.precision,
        )(out)

        if residual.shape != out.shape:
            residual = ConvLayer(
                self.conv_type,
                features=self.features,
                kernel_size=(1, 1),
                strides=1,
                kernel_init=self.kernel_init,
                name="residual_conv",
                dtype=self.dtype,
                precision=self.precision,
            )(residual)

        out = out + residual

        out = (
            jnp.concatenate([out, extra_features], axis=-1)
            if extra_features is not None
            else out
        )

        return out



class UNet(nn.Module):
    out_channels: int = 1
    emb_features: int = 64 * 2
    feature_depths: list = field(default_factory=lambda: [64, 128, 256, 512])
    num_res_blocks: int = 2
    num_middle_res_blocks: int = 1
    activation: Callable = jax.nn.swish
    norm_groups: int = 8
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    kernel_init: Callable = partial(kernel_init, dtype=jnp.float32)
    reshape:bool =True

    def setup(self):
        if self.norm_groups > 0:
            norm = partial(nn.GroupNorm, self.norm_groups)
            self.conv_out_norm = norm()
        else:
            norm = partial(nn.RMSNorm, 1e-5)
            self.conv_out_norm = norm()

    @nn.compact
    def __call__(self, x, conditional):

        input_x = x
        if self.reshape:
            xy = x.shape[-1]
            nn = int(xy**0.5)
            x = rearrange(x, '(X Y) -> X Y', X=nn)

        # add batch dims and channel dim
        x = x[None, ..., None]
        conditional = conditional[None]
        cond = TimeProjection(features=self.emb_features)(conditional)


        feature_depths = self.feature_depths
        conv_type = "conv"
        x = ConvLayer(
            conv_type,
            features=self.feature_depths[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=self.kernel_init(scale=1.0),
            dtype=self.dtype,
            precision=self.precision,
        )(x)
        downs = [x]



        # Downscaling blocks
        for i, dim_out in enumerate(feature_depths):

            dim_in = x.shape[-1]
            # dim_in = dim_out
            for j in range(self.num_res_blocks):

                x = ResidualBlock(
                    conv_type,
                    name=f"down_{i}_residual_{j}",
                    features=dim_in,
                    kernel_init=self.kernel_init(scale=1.0),
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation=self.activation,
                    norm_groups=self.norm_groups,
                    dtype=self.dtype,
                    precision=self.precision,
                )(x, cond)

                downs.append(x)
            if i != len(feature_depths) - 1:
                # print("Downsample", i, x.shape)
                x = Downsample(
                    features=dim_out,
                    scale=2,
                    activation=self.activation,
                    name=f"down_{i}_downsample",
                    dtype=self.dtype,
                    precision=self.precision,
                )(x)

        # Middle Blocks
        middle_dim_out = self.feature_depths[-1]
        for j in range(self.num_middle_res_blocks):
            x = ResidualBlock(
                conv_type,
                name=f"middle_res1_{j}",
                features=middle_dim_out,
                kernel_init=self.kernel_init(scale=1.0),
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                norm_groups=self.norm_groups,
                dtype=self.dtype,
                precision=self.precision,
            )(x, cond)

            x = ResidualBlock(
                conv_type,
                name=f"middle_res2_{j}",
                features=middle_dim_out,
                kernel_init=self.kernel_init(scale=1.0),
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                norm_groups=self.norm_groups,
                dtype=self.dtype,
                precision=self.precision,
            )(x, cond)

        # Upscaling Blocks
        for i, dim_out in enumerate(reversed(feature_depths)):
            for j in range(self.num_res_blocks):
                x = jnp.concatenate([x, downs.pop()], axis=-1)
                kernel_size = (3, 3)
                x = ResidualBlock(
                    conv_type,
                    name=f"up_{i}_residual_{j}",
                    features=dim_out,
                    kernel_init=self.kernel_init(scale=1.0),
                    kernel_size=kernel_size,
                    strides=(1, 1),
                    activation=self.activation,
                    norm_groups=self.norm_groups,
                    dtype=self.dtype,
                    precision=self.precision,
                )(x, cond)
   
            if i != len(feature_depths) - 1:
                x = Upsample(
                    features=feature_depths[-i],
                    scale=2,
                    activation=self.activation,
                    name=f"up_{i}_upsample",
                    dtype=self.dtype,
                    precision=self.precision,
                )(x)

        x = ConvLayer(
            conv_type,
            features=self.feature_depths[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=self.kernel_init(scale=1.0),
            dtype=self.dtype,
            precision=self.precision,
        )(x)


        x = jnp.concatenate([x, downs.pop()], axis=-1)

        x = ResidualBlock(
            conv_type,
            name="final_residual",
            features=self.feature_depths[0],
            kernel_init=self.kernel_init(scale=1.0),
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=self.activation,
            norm_groups=self.norm_groups,
            dtype=self.dtype,
            precision=self.precision,
        )(x, cond)

        x = self.conv_out_norm(x)
        x = self.activation(x)

        x = ConvLayer(
            conv_type,
            features=self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=self.kernel_init(scale=0.0),
            dtype=self.dtype,
            precision=self.precision,
        )(x)


        x = x.reshape(-1)
        input_x = input_x.reshape(-1)

        out = jnp.dot(x, input_x)

        return out
