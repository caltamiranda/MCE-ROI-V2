# rf_pipeline/data_loader.py

import zarr
import numpy as np
from typing import Generator, Optional


class IQDataLoader:
    """
    I/Q signal loader for TorchSig-style .zarr datasets.
    - Lazy loading for memory efficiency.
    - Supports sequential streaming (ideal for edge inference / low RAM).
    """

    def __init__(self, zarr_path: str):
        """
        Args:
            zarr_path (str): Absolute or relative path to the .zarr dataset.
        """
        self.zarr_path = zarr_path
        self.dataset: Optional[zarr.core.Array] = None

    def load_dataset(self) -> None:
        """Explicit lazy load. Called automatically before first access."""
        print(f"[INFO] Loading dataset from: {self.zarr_path}")
        self.dataset = zarr.open(self.zarr_path, mode='r')
        print(f"[OK] Dataset loaded. Total samples: {self.dataset.shape[0]}")

    def stream_signals(self) -> Generator[np.ndarray, None, None]:
        """
        Yields I/Q signals one by one (1D complex numpy arrays).
        Prevents loading the full dataset into RAM at once.
        """
        if self.dataset is None:
            self.load_dataset()

        for idx in range(self.dataset.shape[0]):
            yield self.dataset[idx]  # Complex64 or Complex128 (TorchSig format)

    def get_signal_by_index(self, idx: int) -> np.ndarray:
        """
        Random-access retrieval of a specific I/Q signal.

        Args:
            idx (int): Row index inside the .zarr dataset.

        Returns:
            np.ndarray: 1D complex array representing the I/Q signal.
        """
        if self.dataset is None:
            self.load_dataset()
        return self.dataset[idx]


# Debug / quick test (not used in pipeline)
if __name__ == "__main__":
    loader = IQDataLoader(
        r"C:\Users\User\Documents\Git\torchsig\dataset\torchsig_narrowband_impaired\data\0000000000.zarr"
    )

    for iq in loader.stream_signals():
        print(f"Sample I/Q shape: {iq.shape}, dtype: {iq.dtype}")
        break  # Only show the first one
