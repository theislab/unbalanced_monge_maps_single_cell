import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence


@jax.jit
def sinkhorn_loss(
    x: jnp.ndarray, y: jnp.ndarray, epsilon: float = 0.1, power: float = 2.0
) -> float:
    """Compute transport between (x, a) and (y, b) via Sinkhorn algorithm."""
    a = jnp.ones(len(x)) / len(x)
    b = jnp.ones(len(y)) / len(y)

    sdiv = sinkhorn_divergence(
        pointcloud.PointCloud, x, y, power=power, epsilon=epsilon, a=a, b=b
    )
    return sdiv.divergence
