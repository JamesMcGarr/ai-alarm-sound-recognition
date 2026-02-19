"""
tests/test_dataset.py
---------------------
Unit tests for AlarmDataset and related helpers.
Exercises .wav loading, label assignment, and train/val splitting
using synthetic fixture files – no USB microphone required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile as wavfile
import torch

from src.audio.features import SAMPLE_RATE
from src.training.dataset import AlarmDataset, make_splits, _load_wav_float32


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_wav(path: Path, duration: float = 1.0, freq: float = 1000.0) -> None:
    """Write a synthetic sine-wave .wav to *path*."""
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    wavfile.write(str(path), SAMPLE_RATE, audio)


@pytest.fixture()
def sample_dirs(tmp_path: Path):
    """Create temporary positive/ and negative/ dirs with synthetic .wav files."""
    pos_dir = tmp_path / "positive"
    neg_dir = tmp_path / "negative"
    pos_dir.mkdir()
    neg_dir.mkdir()

    for i in range(10):
        _write_wav(pos_dir / f"positive_{i:04d}.wav", freq=3000.0)
    for i in range(10):
        _write_wav(neg_dir / f"negative_{i:04d}.wav", freq=500.0)

    return pos_dir, neg_dir


# ── tests ─────────────────────────────────────────────────────────────────────

class TestLoadWavFloat32:
    def test_loads_int16(self, tmp_path: Path):
        path = tmp_path / "test.wav"
        _write_wav(path)
        audio = _load_wav_float32(path)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert -1.0 <= audio.max() <= 1.0

    def test_length_matches_duration(self, tmp_path: Path):
        path = tmp_path / "test.wav"
        _write_wav(path, duration=1.0)
        audio = _load_wav_float32(path)
        assert len(audio) == SAMPLE_RATE


class TestAlarmDataset:
    def test_length(self, sample_dirs):
        pos_dir, neg_dir = sample_dirs
        ds = AlarmDataset(positive_dir=pos_dir, negative_dir=neg_dir)
        assert len(ds) == 20

    def test_label_balance(self, sample_dirs):
        pos_dir, neg_dir = sample_dirs
        ds = AlarmDataset(positive_dir=pos_dir, negative_dir=neg_dir)
        assert ds.n_positive == 10
        assert ds.n_negative == 10

    def test_item_shapes(self, sample_dirs):
        pos_dir, neg_dir = sample_dirs
        ds = AlarmDataset(positive_dir=pos_dir, negative_dir=neg_dir)
        spec, label = ds[0]
        assert spec.ndim == 3         # (1, N_MELS, T)
        assert spec.shape[0] == 1
        assert label.ndim == 0        # scalar
        assert label.dtype == torch.float32

    def test_labels_are_binary(self, sample_dirs):
        pos_dir, neg_dir = sample_dirs
        ds = AlarmDataset(positive_dir=pos_dir, negative_dir=neg_dir)
        labels = {ds[i][1].item() for i in range(len(ds))}
        assert labels == {0.0, 1.0}

    def test_missing_positive_dir_raises(self, tmp_path: Path):
        neg_dir = tmp_path / "negative"
        neg_dir.mkdir()
        _write_wav(neg_dir / "negative_0000.wav")
        with pytest.raises(FileNotFoundError):
            AlarmDataset(positive_dir=tmp_path / "positive", negative_dir=neg_dir)

    def test_missing_negative_dir_raises(self, tmp_path: Path):
        pos_dir = tmp_path / "positive"
        pos_dir.mkdir()
        _write_wav(pos_dir / "positive_0000.wav")
        with pytest.raises(FileNotFoundError):
            AlarmDataset(positive_dir=pos_dir, negative_dir=tmp_path / "negative")


class TestMakeSplits:
    def test_split_sizes(self, sample_dirs):
        pos_dir, neg_dir = sample_dirs
        ds = AlarmDataset(positive_dir=pos_dir, negative_dir=neg_dir)
        train_ds, val_ds = make_splits(ds, train_fraction=0.8)
        assert len(train_ds) == 16
        assert len(val_ds) == 4

    def test_no_overlap(self, sample_dirs):
        pos_dir, neg_dir = sample_dirs
        ds = AlarmDataset(positive_dir=pos_dir, negative_dir=neg_dir)
        train_ds, val_ds = make_splits(ds, train_fraction=0.8)
        train_indices = set(train_ds.indices)
        val_indices = set(val_ds.indices)
        assert train_indices.isdisjoint(val_indices)
