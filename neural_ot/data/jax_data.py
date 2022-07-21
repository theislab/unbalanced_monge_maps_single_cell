from dataclasses import dataclass
from typing import List, Tuple

import jax
import jax.numpy as jnp


@dataclass
class JaxSampler:
    """Data sampler for Jax."""

    data: jnp.float64
    batch_size: int = 128
    shuffle: bool = True
    drop_last: bool = True

    def __post_init__(self):
        self.data_length = len(self.data)

        @jax.jit
        def _sample(key: jax.random.KeyArray) -> jnp.ndarray:
            """Jitted sample function."""
            indeces = jax.random.randint(key, shape=[self.batch_size], minval=0, maxval=self.data_length)
            return self.data[indeces]

        self._sample = _sample

    def __call__(self, key: jax.random.KeyArray, full_dataset: bool = False) -> jnp.ndarray:
        """Sample data."""
        if full_dataset:
            return self.data
        else:
            return self._sample(key)


@dataclass
class PairSampler:
    """Pair data sampler that takes two ordered JaxSamplers and samples in joint pairs."""

    source_samplers: List[JaxSampler]
    target_samplers: List[JaxSampler]
    joint_pairs: bool = True

    def __post_init__(self):
        assert len(self.source_samplers) == len(self.target_samplers)
        self.num_samplers = len(self.source_samplers)

        @jax.jit
        def _sample(key: jax.random.KeyArray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Jitted sample function."""
            index = jax.random.randint(key, shape=[1], minval=0, maxval=self.num_samplers)
            new_key, key = jax.random.split(key, 2)
            if self.joint_pairs:
                return self.source_samplers[index[0]](key), self.target_samplers[index[0]](new_key)
            else:
                target_index = jax.random.randint(new_key, shape=[1], minval=0, maxval=self.num_samplers)
                return self.source_samplers[index[0]](key), self.target_samplers[target_index[0]](new_key)

        self._sample = _sample

    def __call__(self, key: jax.random.KeyArray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample data."""
        return self._sample(key)
