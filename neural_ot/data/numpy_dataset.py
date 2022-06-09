import numpy as np
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    """Basic dataset for numpy arrays."""

    def __init__(self, numpy_data: np.ndarray):
        self.numpy_data = numpy_data

    def __len__(self) -> int:
        return len(self.numpy_data)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.numpy_data[index]
