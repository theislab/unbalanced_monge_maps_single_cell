from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import ott


@dataclass
class JaxSampler:
    """Data sampler for Jax with optional weighting."""

    data: jnp.ndarray
    batch_size: int = 128
    weighting: Optional[jnp.ndarray] = None

    def __post_init__(self):
        # Weighting needs to have the same length as data.
        if self.weighting is not None:
            assert self.data.shape[0] == self.weighting.shape[0]

        @jax.jit
        def _sample(key: jax.random.KeyArray) -> jnp.ndarray:
            """Jitted sample function."""
            if self.weighting is None:
                indeces = jax.random.randint(key, shape=[self.batch_size], minval=0, maxval=self.data_length)
                return self.data[indeces]
            else:
                return jax.random.choice(key, self.data, shape=[self.batch_size], p=self.weighting)

        self._sample = _sample

    def __call__(self, key: jax.random.KeyArray, full_dataset: bool = False) -> jnp.ndarray:
        """Sample data."""
        if full_dataset:
            return self.data
        else:
            return self._sample(key)


@dataclass
class MatchingPairSampler:
    """Data sampler for Jax."""

    data_source: jnp.ndarray
    data_target: jnp.ndarray
    batch_size: int = 128
    tau_a: float = 1.0
    tau_b: float = 1.0
    epsilon: float = 1e-1
    matching: bool = True
    weighting: Optional[jnp.ndarray] = None

    def __post_init__(self):
        self.length_source = len(self.data_source)
        self.length_target = len(self.data_target)
        self.ott_scaling = jnp.ones(self.batch_size) / self.batch_size

        @jax.jit
        def _sample(key: jax.random.KeyArray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Jitted sample function."""
            key_source, key_target, key_indices = jax.random.split(key, 3)
            if self.weighting is None:
                batch_source = self.data_source[
                    jax.random.randint(key_source, shape=[self.batch_size], minval=0, maxval=self.length_source)
                ]
            else:
                batch_source = jax.random.choice(key, self.data_source, shape=[self.batch_size], p=self.weighting)
            batch_target = self.data_target[
                jax.random.randint(key_target, shape=[self.batch_size], minval=0, maxval=self.length_target)
            ]
            if not self.matching:
                return batch_source, batch_target
            # solve regularized ot between batch_source and batch_target
            geom = ott.geometry.pointcloud.PointCloud(
                batch_source, batch_target, epsilon=self.epsilon, scale_cost="mean"
            )
            out = ott.core.sinkhorn.sinkhorn(
                geom,
                self.ott_scaling,
                self.ott_scaling,
                tau_a=self.tau_a,
                tau_b=self.tau_b,
                jit=False,
                max_iterations=1e7,
            )
            # get flattened log transition matrix
            transition_matrix = jnp.log(geom.transport_from_potentials(out.f, out.g).flatten())
            # sample from transition_matrix
            indeces = jax.random.categorical(key_indices, transition_matrix, shape=[self.batch_size])
            indeces_source = indeces // self.batch_size
            indeces_target = indeces % self.batch_size
            return batch_source[indeces_source], batch_target[indeces_target]

        self._sample = _sample

    def __call__(self, key: jax.random.KeyArray, full_dataset: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample data."""
        if full_dataset:
            return self.data_source, self.data_target
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
