from typing import Callable, Sequence

import jax.numpy as jnp
from flax import linen as nn
from ott.core.icnn import PositiveDense


class ICNN(nn.Module):
    """Input convex neural network (ICNN) architecture."""

    dim_hidden: Sequence[int]
    init_std: float = 0.1
    init_fn: Callable = nn.initializers.normal
    act_fn: Callable = nn.leaky_relu
    pos_weights: bool = False
    first_quadratic: bool = True
    last_quadratic_term: bool = False
    quad_rank: int = 3

    def setup(self):
        """Initialize ICNN architecture."""
        num_hidden = len(self.dim_hidden)

        w_zs = []

        if self.pos_weights:
            Dense = PositiveDense
        else:
            Dense = nn.Dense

        for i in range(1, num_hidden):
            w_zs.append(Dense(self.dim_hidden[i], kernel_init=self.init_fn(self.init_std), use_bias=False))
        w_zs.append(Dense(1, kernel_init=self.init_fn(self.init_std), use_bias=False))
        self.w_zs = w_zs

        w_xs = []
        for i in range(num_hidden):
            w_xs.append(nn.Dense(self.dim_hidden[i], kernel_init=self.init_fn(self.init_std), use_bias=True))
        w_xs.append(nn.Dense(1, kernel_init=self.init_fn(self.init_std), use_bias=True))
        self.w_xs = w_xs

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply ICNN module."""
        z = self.act_fn(self.w_xs[0](x))
        if self.first_quadratic:
            z = jnp.multiply(z, z)

        for Wz, Wx in zip(self.w_zs[:-1], self.w_xs[1:-1]):
            z = self.act_fn(jnp.add(Wz(z), Wx(x)))
        y = jnp.add(self.w_zs[-1](z), self.w_xs[-1](x))
        if self.last_quadratic_term:
            L = self.param("L", nn.initializers.normal(), (self.quad_rank, x.shape[-1]))
            quad = x.dot(L.transpose().dot(L)).dot(x.transpose())
            y += quad
        return jnp.mean(y, axis=-1)
