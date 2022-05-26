from typing import Callable, Sequence

import jax
from flax import linen as nn
from ott.core import icnn


def get_icnn(
    dim_hidden: Sequence[int],
    init_std: float = 0.1,
    init_fn: Callable = jax.nn.initializers.normal,
    act_fn: Callable = nn.leaky_relu,
    pos_weights: bool = True,
) -> icnn.ICNN:
    """Return an instance of the ICNN model."""
    return icnn.ICNN(dim_hidden, init_std, init_fn, act_fn, pos_weights)
