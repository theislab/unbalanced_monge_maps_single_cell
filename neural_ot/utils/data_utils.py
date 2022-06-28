from typing import Optional, Tuple

import numpy as np
import scanpy as sc
from data import NumpyDataset
from torch.utils.data import DataLoader


def get_anndata_loaders(
    rng: np.random.Generator, adata_path: str, use_pca: bool, batch_size: int = 64, full_dataset: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load AnnData object, split data and return a tuple of data loaders."""
    adata = sc.read(adata_path, backed="r+", cache=True)
    if full_dataset:
        if use_pca:
            data = adata.obsm["X_pca"]
        else:
            data = np.log(adata.X.toarray() + 1)
        dataset = NumpyDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        return dataloader, dataloader, dataloader
    # split data
    indices = np.arange(len(adata))
    rng.shuffle(indices)
    train_indices = indices[: int(np.round(len(adata) * 0.8))]
    val_indices = indices[int(np.round(len(adata) * 0.8)) : int(np.round(len(adata) * 0.9))]
    test_indices = indices[int(np.round(len(adata) * 0.9)) :]
    if use_pca:
        train_data = adata.obsm["X_pca"][train_indices]
        val_data = adata.obsm["X_pca"][val_indices]
        test_data = adata.obsm["X_pca"][test_indices]
    else:
        train_data = np.log(adata.X[train_indices].toarray() + 1)
        val_data = np.log(adata.X[val_indices].toarray() + 1)
        test_data = np.log(adata.X[test_indices].toarray() + 1)
    # create datasets & loaders
    train_dataset = NumpyDataset(train_data)
    valid_dataset = NumpyDataset(val_data)
    test_dataset = NumpyDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, valid_loader, test_loader


def get_gaussian_mixture_loaders(
    rng: np.random.Generator,
    input_dim: int,
    centers: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    num_centers: int = 1,
    num_samples: int = 15000,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Generate Gaussian mixture data, split data and return a tuple of data loaders."""
    # create default centers and sigma
    if centers is None:
        centers = np.array(
            [[np.sqrt(input_dim)] + [1.0 + center * 10.0] * (input_dim - 1) for center in np.arange(num_centers)]
        )
    if sigma is None:
        sigma = np.eye(input_dim) * input_dim
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
