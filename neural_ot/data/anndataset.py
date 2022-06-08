import numpy as np
import scanpy as sc
from torch.utils.data import Dataset


class AnnDataset(Dataset):
    """
    Torch datast for AnnData objects including automatic data spltting.

    Args:
        adata_path: Path to AnnData object.
        split: Split to use.
        rng: Random number generator.
        use_pca: Whether to use PCA embedding.
    """

    def __init__(self, adata_path: str, split: str, rng: np.random.Generator, use_pca: bool = False):
        self.adata = sc.read_h5ad(adata_path)
        self.use_pca = use_pca
        # split data
        indices = np.arange(self.__len__())
        rng.shuffle(indices)
        train_indeces = indices[int(np.round(len(self.adata) * 0.8)) :]
        val_indeces = indices[int(np.round(len(self.adata) * 0.8)) : int(np.round(len(self.adata) * 0.9))]
        test_indices = indices[int(np.round(len(self.adata) * 0.9)) :]
        # testing
        match split:
            case "train":
                self.indices = train_indeces
            case "val":
                self.indices = val_indeces
            case "test":
                self.indices = test_indices
            case _:
                raise ValueError(f"Unknown split: {split}")
        self.adata = self.adata[self.indices]

    def __len__(self) -> int:
        return len(self.adata)

    def __getitem__(self, index: int) -> np.ndarray:
        if self.use_pca:
            return self.adata[index].obsm["X_pca"].toarray().squeeze()
        else:
            return self.adata[index].X.toarray().squeeze()
