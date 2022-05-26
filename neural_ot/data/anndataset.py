from typing import Iterator

import anndata
import jax.numpy as jnp
import numpy as np
import scanpy as sc
from torch.utils.data import IterableDataset


class AnnDataset(IterableDataset):
    """
    Torch datast for AnnData objects.

    Args:
        adata_path: Path to AnnData object.
    """

    def __init__(self, adata_path: str):
        self.adata: anndata.AnnData = sc.read(adata_path)

    def __len__(self) -> int:
        return len(self.adata)

    def __iter__(self) -> Iterator[jnp.ndarray]:
        while True:
            yield jnp.array(
                self.adata[np.random.choice(self.__len__())].X.toarray().squeeze()
            )
