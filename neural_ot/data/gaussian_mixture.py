import numpy as np
from torch.utils.data import Dataset


class GaussianMixtureDataset(Dataset):
    """
    Dataset generated from N-dimensional multivariate Gaussians with M centers.

    Args:
        centers: M centers of the Gaussian distributions, shape [M, N]
        sigma: Standard deviation for all Gaussian distributions, shape [N, N]
        n_samples: number of samples to generate (default: 10000)
    """

    def __init__(
        self, centers: np.ndarray, sigma: np.ndarray, split: str, rng: np.random.Generator, num_samples: int = 10000
    ):
        self.centers = centers
        self.sigma = sigma
        self.num_samples = num_samples
        self.input_dim = sigma.shape[0]
        # generate data
        cholesky_l = np.linalg.cholesky(sigma + 1e-4 * np.eye(self.input_dim))
        u = rng.normal(loc=0, scale=1, size=[self.num_samples, self.input_dim])
        center = self.centers[rng.choice(len(self.centers), size=self.num_samples), :]
        self.data = center + np.tensordot(u, cholesky_l, axes=1)
        # split data
        indices = np.arange(len(self.data))
        rng.shuffle(indices)
        train_indeces = indices[: int(np.round(len(self.data) * 0.8))]
        val_indeces = indices[int(np.round(len(self.data) * 0.8)) : int(np.round(len(self.data) * 0.9))]
        test_indices = indices[int(np.round(len(self.data) * 0.9)) :]
        match split:
            case "train":
                self.indices = train_indeces
            case "val":
                self.indices = val_indeces
            case "test":
                self.indices = test_indices
            case _:
                raise ValueError(f"Unknown split: {split}")
        self.data = self.data[self.indices]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.data[index]
