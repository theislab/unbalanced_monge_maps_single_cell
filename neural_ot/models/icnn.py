from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from ott.core.icnn import PosDefPotentials, PositiveDense
from ott.geometry.matrix_square_root import sqrtm, sqrtm_only


class ICNN(nn.Module):
    """Input convex neural network (ICNN) architecture."""

    dim_hidden: Sequence[int]
    init_std: float = 0.1
    init_fn: Callable = nn.initializers.normal
    act_fn: Callable = nn.leaky_relu
    pos_weights: bool = False
    first_quadratic: bool = True
    first_quadratic_term: bool = False
    last_quadratic_term: bool = False
    quad_rank: Optional[int] = 3
    dim_data: int = 2
    closed_form_init: bool = False
    gaussian_map: Tuple[jnp.ndarray, jnp.ndarray] = None

    def setup(self):
        """Initialize ICNN architecture."""
        num_hidden = len(self.dim_hidden)

        if self.pos_weights:
            Dense = PositiveDense
            # this function needs to be the inverse map of function
            # used in PositiveDense layers
            rescale = Dense.inv_rectifier_fn
        else:
            Dense = nn.Dense
            rescale = lambda x: x

        if self.closed_form_init:
            assert self.first_quadratic_term
            if self.gaussian_map is not None:
                factor, mean = self.compute_gaussian_map(self.gaussian_map)
            else:
                factor, mean = self.compute_identity_map(self.dim_data)
            kernel_inits_wz = [nn.initializers.constant(rescale(1.0 / dim)) for dim in self.dim_hidden]
            kernel_inits_wz.insert(0, nn.initializers.constant(rescale(1.0)))
            kernel_init_wx = nn.initializers.constant(0.0)
            bias_init = nn.initializers.constant(1.0)
            kernel_init_wx_pot = lambda *args, **kwargs: factor
            bias_init_wx_pot = lambda *args, **kwargs: mean
        else:
            kernel_inits_wz = self.init_fn(self.init_std)
            kernel_init_wx = self.init_fn(self.init_std)
            bias_init = nn.initializers.constant(0.0)
            kernel_init_wx_pot = nn.initializers.lecun_normal()
            bias_init_wx_pot = nn.initializers.zeros

        w_zs = []
        for i in range(1, num_hidden):
            w_zs.append(
                Dense(
                    self.dim_hidden[i],
                    kernel_init=kernel_inits_wz[i],
                    use_bias=False,
                )
            )
        w_zs.append(Dense(1, kernel_init=kernel_inits_wz[-1], use_bias=False))
        self.w_zs = w_zs

        w_xs = []
        loop_start = 0
        if self.first_quadratic_term:
            # increase loop start by one because first layer already added
            loop_start += 1
            w_xs.append(
                PosDefPotentials(
                    self.dim_data,
                    num_potentials=1,
                    kernel_init=kernel_init_wx_pot,
                    bias_init=bias_init_wx_pot,
                    use_bias=True,
                )
            )
        for i in range(loop_start, num_hidden):
            w_xs.append(
                nn.Dense(
                    self.dim_hidden[i],
                    kernel_init=kernel_init_wx,
                    bias_init=bias_init,
                    use_bias=True,
                )
            )
        w_xs.append(
            nn.Dense(
                1,
                kernel_init=kernel_init_wx,
                bias_init=bias_init,
                use_bias=True,
            )
        )
        self.w_xs = w_xs

    def compute_gaussian_map(self, inputs):
        """Compute Gaussian map from the inputs."""

        def compute_moments(x, reg=1e-4, sqrt_inv=False):
            """Compute mean and covariance matrix of the Gaussian."""
            shape = x.shape
            z = x.reshape(shape[0], -1)
            mu = jnp.expand_dims(jnp.mean(z, axis=0), 0)
            z = z - mu
            matmul = lambda a, b: jnp.matmul(a, b)
            sigma = jax.vmap(matmul)(jnp.expand_dims(z, 2), jnp.expand_dims(z, 1))
            # unbiased estimate
            sigma = jnp.sum(sigma, axis=0) / (shape[0] - 1)
            # regularize
            sigma = sigma + reg * jnp.eye(shape[1])

            if sqrt_inv:
                sigma_sqrt, sigma_inv_sqrt, _ = sqrtm(sigma)
                return sigma, sigma_sqrt, sigma_inv_sqrt, mu
            else:
                return sigma, mu

        source, target = inputs
        _, covs_sqrt, covs_inv_sqrt, mus = compute_moments(source, sqrt_inv=True)
        covt, mut = compute_moments(target, sqrt_inv=False)

        mo = sqrtm_only(jnp.dot(jnp.dot(covs_sqrt, covt), covs_sqrt))
        A = jnp.dot(jnp.dot(covs_inv_sqrt, mo), covs_inv_sqrt)
        b = jnp.squeeze(mus) - jnp.linalg.solve(A, jnp.squeeze(mut))
        A = sqrtm_only(A)

        return jnp.expand_dims(A, 0), jnp.expand_dims(b, 0)

    def compute_identity_map(self, input_dim):
        """Compute identity map from the inputs."""
        A = jnp.eye(input_dim).reshape((1, input_dim, input_dim))
        b = jnp.zeros((1, input_dim))

        return A, b

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply ICNN module."""
        z = self.w_xs[0](x)
        if not self.first_quadratic_term:
            z = self.act_fn(z)
        if self.first_quadratic:
            z = jnp.multiply(z, z)
        for Wz, Wx in zip(self.w_zs[:-1], self.w_xs[1:-1]):
            z = self.act_fn(jnp.add(Wz(z), Wx(x)))
        y = jnp.add(self.w_zs[-1](z), self.w_xs[-1](x))
        if self.last_quadratic_term:
            L = self.param("L", nn.initializers.lecun_normal(), (self.quad_rank, x.shape[-1]))
            quad = x.dot(L.transpose().dot(L)).dot(x.transpose())
            y += quad
        return y.squeeze()
