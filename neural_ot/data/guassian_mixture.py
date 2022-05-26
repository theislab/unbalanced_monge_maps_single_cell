from typing import Iterator

import jax.numpy as jnp
import numpy as np
from torch.utils.data import IterableDataset


class GaussianMixtureDataset(IterableDataset):
    """
    Dataset generated from N-dimensional multivariate Gaussians with M centers.

    Args:
        centers: M centers of the Gaussian distributions, shape [M, N]
        sigma: Standard deviation for all Gaussian distributions, shape [N, N]
        n_samples: number of samples to generate (default: 10000)
    """

    def __init__(self, centers: np.ndarray, sigma: np.ndarray, n_samples: int = 10000):
        self.centers = centers
        self.sigma = sigma
        self.n_samples = n_samples
        self.input_dim = sigma.shape[0]
        self.cholesky_l = np.linalg.cholesky(sigma + 1e-4 * np.eye(self.input_dim))

    def __len__(self) -> int:
        return self.n_samples

    def __iter__(self) -> Iterator[jnp.ndarray]:
        while True:
            center = self.centers[np.random.choice(len(self.centers))]
            u = np.random.normal(loc=0, scale=1, size=self.input_dim)
            yield jnp.array(center + np.dot(self.cholesky_l, u))
