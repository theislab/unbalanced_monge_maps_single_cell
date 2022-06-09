from typing import Tuple

import numpy as np
import scanpy as sc
from data import NumpyDataset
from torch.utils.data import DataLoader


def get_anndata_loaders(
    adata_path: str, use_pca: bool = False, batch_size: int = 32, rng: np.random.Generator = np.random.default_rng()
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
        train_data = adata.X[train_indeces]
        val_data = adata.X[val_indeces]
        test_data = adata.X[test_indices]
    # create datasets & loaders
    train_dataset = NumpyDataset(train_data)
    val_dataset = NumpyDataset(val_data)
    test_dataset = NumpyDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader
