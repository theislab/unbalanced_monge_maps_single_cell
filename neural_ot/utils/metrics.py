import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence


@jax.jit
def sinkhorn_loss(x: jnp.ndarray, y: jnp.ndarray, epsilon: float = 0.1, power: float = 2.0) -> float:
    """Compute transport between (x, a) and (y, b) via Sinkhorn algorithm."""
    a = jnp.ones(len(x)) / len(x)
    b = jnp.ones(len(y)) / len(y)

    sdiv = sinkhorn_divergence(pointcloud.PointCloud, x, y, power=power, epsilon=epsilon, a=a, b=b)
    return sdiv.divergence


@jax.jit
def mmd_rbf(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute MMD between x and y via RBF kernel."""
    x_norm = jnp.square(x).sum(-1)
    xx = jnp.einsum("ia, ja- > ij", x, x)
    x_sq_dist = x_norm[..., :, None] + x_norm[..., None, :] - 2 * xx

    y_norm = jnp.square(y).sum(-1)
    yy = jnp.einsum("ia, ja -> ij", y, y)
    y_sq_dist = y_norm[..., :, None] + y_norm[..., None, :] - 2 * yy

    zz = jnp.einsum("ia, ja -> ij", x, y)
    z_sq_dist = x_norm[..., :, None] + y_norm[..., None, :] - 2 * zz

    var = jnp.var(z_sq_dist)
    XX, YY, XY = (jnp.zeros(xx.shape), jnp.zeros(yy.shape), jnp.zeros(zz.shape))

    bandwidth_range = [0.5, 0.1, 0.01, 0.005]
    for scale in bandwidth_range:
        XX += jnp.exp(-0.5 * x_sq_dist / (var * scale))
        YY += jnp.exp(-0.5 * y_sq_dist / (var * scale))
        XY += jnp.exp(-0.5 * z_sq_dist / (var * scale))

    return jnp.mean(XX) + jnp.mean(YY) - 2.0 * jnp.mean(XY)


@jax.jit
def mmd_linear(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute MMD between x and y via linear kernel."""
    delta = x.mean(0) - y.mean(0)
    return delta.dot(delta.T)
