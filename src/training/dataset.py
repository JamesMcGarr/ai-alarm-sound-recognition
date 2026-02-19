"""
training/dataset.py
-------------------
PyTorch Dataset that loads labelled .wav files from the data/ directory and
converts each clip to a log-mel-spectrogram tensor via audio/features.py.

Label convention
    1  – alarm (positive)   ← data/positive/
    0  – not alarm (negative) ← data/negative/
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io.wavfile as wavfile
import torch
from torch.utils.data import Dataset, random_split

import src.audio.features as F

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
POSITIVE_DIR = DATA_DIR / "positive"
NEGATIVE_DIR = DATA_DIR / "negative"


def _load_wav_float32(path: Path) -> np.ndarray:
    """Load a .wav file as a normalised float32 numpy array."""
    rate, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data[:, 0]   # keep first channel only
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    return data


class AlarmDataset(Dataset):
    """
    Dataset built from .wav files in ``data/positive/`` and ``data/negative/``.

    Each item is a tuple ``(spectrogram_tensor, label_tensor)`` where:
    - ``spectrogram_tensor`` has shape ``(1, N_MELS, time_frames)``
    - ``label_tensor`` is a scalar float32 tensor (0.0 or 1.0)
    """

    def __init__(
        self,
        positive_dir: Path = POSITIVE_DIR,
        negative_dir: Path = NEGATIVE_DIR,
        augment: bool = False,
    ) -> None:
        self.augment = augment
        self._items: list[Tuple[Path, int]] = []

        pos_files = sorted(positive_dir.glob("*.wav"))
        neg_files = sorted(negative_dir.glob("*.wav"))

        if not pos_files:
            raise FileNotFoundError(
                f"No .wav files found in {positive_dir}. "
                "Run the collector first: python train.py --collect"
            )
        if not neg_files:
            raise FileNotFoundError(
                f"No .wav files found in {negative_dir}. "
                "Run the collector first: python train.py --collect"
            )

        self._items.extend((p, 1) for p in pos_files)
        self._items.extend((p, 0) for p in neg_files)

        random.shuffle(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self._items[idx]
        audio = _load_wav_float32(path)

        if self.augment:
            audio = _augment(audio)

        tensor = F.extract(audio)                          # (1, N_MELS, T)
        return tensor, torch.tensor(label, dtype=torch.float32)

    # ── convenience ───────────────────────────────────────────────────────────

    @property
    def n_positive(self) -> int:
        return sum(1 for _, l in self._items if l == 1)

    @property
    def n_negative(self) -> int:
        return sum(1 for _, l in self._items if l == 0)

    def summary(self) -> str:
        return (
            f"AlarmDataset: {len(self)} samples  "
            f"(pos={self.n_positive}, neg={self.n_negative})"
        )


def _augment(audio: np.ndarray) -> np.ndarray:
    """Lightweight on-the-fly augmentation for training robustness."""
    # Additive Gaussian noise
    noise_level = random.uniform(0.0, 0.005)
    audio = audio + np.random.randn(*audio.shape).astype(np.float32) * noise_level

    # Random amplitude scaling
    scale = random.uniform(0.7, 1.3)
    audio = (audio * scale).clip(-1.0, 1.0)

    return audio


def make_splits(
    dataset: AlarmDataset,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Split *dataset* into (train_subset, val_subset) using *train_fraction*.

    Parameters
    ----------
    dataset:
        The full AlarmDataset.
    train_fraction:
        Proportion to use for training (rest goes to validation).
    seed:
        Random seed for reproducible splits.

    Returns
    -------
    tuple[Dataset, Dataset]
        ``(train_dataset, val_dataset)``
    """
    n_total = len(dataset)
    n_train = int(n_total * train_fraction)
    n_val = n_total - n_train
    return random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
