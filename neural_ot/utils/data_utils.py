from typing import Tuple

import numpy as np
import scanpy as sc
from data import NumpyDataset
from torch.utils.data import DataLoader


def get_anndata_loaders(
    adata_path: str, use_pca: bool, rng: np.random.Generator, batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load AnnData object, split data and return a tuple of data loaders."""
    adata = sc.read(adata_path, backed="r+", cache=True)
    # split data
    indices = np.arange(len(adata))
    rng.shuffle(indices)
    train_indeces = indices[: int(np.round(len(adata) * 0.8))]
    val_indeces = indices[int(np.round(len(adata) * 0.8)) : int(np.round(len(adata) * 0.9))]
    test_indices = indices[int(np.round(len(adata) * 0.9)) :]
    if use_pca:
        train_data = adata.obsm["X_pca"][train_indeces]
        val_data = adata.obsm["X_pca"][val_indeces]
        test_data = adata.obsm["X_pca"][test_indices]
    else:
        train_data = adata.X[train_indeces].toarray()
        val_data = adata.X[val_indeces].toarray()
        test_data = adata.X[test_indices].toarray()
    # create datasets & loaders
    train_dataset = NumpyDataset(train_data)
    valid_dataset = NumpyDataset(val_data)
    test_dataset = NumpyDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, valid_loader, test_loader


def get_gaussian_mixture_loaders(
    centers: np.ndarray,
    sigma: np.ndarray,
    input_dim: int,
    rng: np.random.Generator,
    num_samples: int = 15000,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Generate Gaussian mixture data, split data and return a tuple of data loaders."""
    # generate data
    cholesky_l = np.linalg.cholesky(sigma + 1e-4 * np.eye(input_dim))
    u = rng.normal(loc=0, scale=1, size=[num_samples, input_dim])
    center = centers[rng.choice(len(centers), size=num_samples), :]
    data = center + np.tensordot(u, cholesky_l, axes=1)
    # split data
    indices = np.arange(len(data))
    rng.shuffle(indices)
    train_data = data[indices[: int(np.round(len(data) * 0.8))]]
    val_data = data[indices[int(np.round(len(data) * 0.8)) : int(np.round(len(data) * 0.9))]]
    test_data = data[indices[int(np.round(len(data) * 0.9)) :]]
    # create datasets & loaders
    train_dataset = NumpyDataset(train_data)
    valid_dataset = NumpyDataset(val_data)
    test_dataset = NumpyDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, valid_loader, test_loader
