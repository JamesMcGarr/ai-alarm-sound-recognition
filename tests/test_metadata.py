"""
tests/test_metadata.py
----------------------
Unit tests for training metadata generation, dated model paths,
symlinks, evaluation metrics, and the file_paths dataset property.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import scipy.io.wavfile as wavfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.audio.features import N_MELS, SAMPLE_RATE
from src.training.dataset import AlarmDataset, make_splits
from src.training.model import AlarmCNN
from src.training.trainer import (
    _build_metadata,
    _dated_model_paths,
    _evaluate,
    _update_symlink,
    _val_sample_counts,
)


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


# ── TestDatedModelPaths ──────────────────────────────────────────────────────

class TestDatedModelPaths:
    def test_returns_dated_names(self, tmp_path: Path):
        model_path, meta_path = _dated_model_paths(tmp_path)
        assert "alarm_model_" in model_path.name
        assert model_path.suffix == ".pt"
        assert meta_path.name == model_path.stem + "_meta.json"

    def test_disambiguation_on_collision(self, tmp_path: Path):
        # Create the first file
        first_model, first_meta = _dated_model_paths(tmp_path)
        first_model.touch()
        first_meta.touch()

        # Second call should return a _2 variant
        second_model, second_meta = _dated_model_paths(tmp_path)
        assert "_2.pt" in second_model.name
        assert "_2_meta.json" in second_meta.name
        assert second_model != first_model

    def test_multiple_disambiguations(self, tmp_path: Path):
        first, _ = _dated_model_paths(tmp_path)
        first.touch()
        second, _ = _dated_model_paths(tmp_path)
        second.touch()
        third, third_meta = _dated_model_paths(tmp_path)
        assert "_3.pt" in third.name
        assert "_3_meta.json" in third_meta.name


# ── TestUpdateSymlink ────────────────────────────────────────────────────────

class TestUpdateSymlink:
    def test_creates_symlink(self, tmp_path: Path):
        target = tmp_path / "alarm_model_2026-03-17.pt"
        target.touch()
        link = tmp_path / "alarm_model.pt"
        _update_symlink(link, target)
        assert link.is_symlink()
        assert link.resolve() == target.resolve()

    def test_replaces_existing_symlink(self, tmp_path: Path):
        old_target = tmp_path / "alarm_model_2026-02-25.pt"
        old_target.touch()
        new_target = tmp_path / "alarm_model_2026-03-17.pt"
        new_target.touch()
        link = tmp_path / "alarm_model.pt"
        _update_symlink(link, old_target)
        _update_symlink(link, new_target)
        assert link.is_symlink()
        assert link.resolve() == new_target.resolve()

    def test_replaces_regular_file(self, tmp_path: Path):
        link = tmp_path / "alarm_model.pt"
        link.write_bytes(b"old model data")
        target = tmp_path / "alarm_model_2026-03-17.pt"
        target.touch()
        _update_symlink(link, target)
        assert link.is_symlink()


# ── TestEvaluate ─────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_returns_expected_keys(self):
        model = AlarmCNN()
        model.eval()
        # Create a single dummy batch
        batch_size = 4
        specs = torch.randn(batch_size, 1, N_MELS, 87)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        dataset = torch.utils.data.TensorDataset(specs, labels)
        loader = DataLoader(dataset, batch_size=batch_size)
        criterion = nn.BCELoss()

        result = _evaluate(model, loader, criterion, torch.device("cpu"))

        expected_keys = {
            "val_loss", "val_accuracy", "precision", "recall",
            "f1", "false_positive_rate", "false_negative_rate",
        }
        assert set(result.keys()) == expected_keys

    def test_values_in_valid_range(self):
        model = AlarmCNN()
        model.eval()
        specs = torch.randn(8, 1, N_MELS, 87)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        dataset = torch.utils.data.TensorDataset(specs, labels)
        loader = DataLoader(dataset, batch_size=8)
        criterion = nn.BCELoss()

        result = _evaluate(model, loader, criterion, torch.device("cpu"))

        for key in ["val_accuracy", "precision", "recall", "f1",
                     "false_positive_rate", "false_negative_rate"]:
            assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of range"
        assert result["val_loss"] >= 0.0


# ── TestValSampleCounts ──────────────────────────────────────────────────────

class TestValSampleCounts:
    def test_counts_correct(self, sample_dirs):
        pos_dir, neg_dir = sample_dirs
        ds = AlarmDataset(positive_dir=pos_dir, negative_dir=neg_dir)
        _, val_ds = make_splits(ds, train_fraction=0.8)
        counts = _val_sample_counts(val_ds, ds)
        assert counts["total"] == len(val_ds)
        assert counts["positive"] + counts["negative"] == counts["total"]
        assert counts["positive"] >= 0
        assert counts["negative"] >= 0


# ── TestFilePathsProperty ────────────────────────────────────────────────────

class TestFilePathsProperty:
    def test_returns_grouped_paths(self, sample_dirs):
        pos_dir, neg_dir = sample_dirs
        ds = AlarmDataset(positive_dir=pos_dir, negative_dir=neg_dir)
        paths = ds.file_paths
        assert "positive" in paths
        assert "negative" in paths
        assert len(paths["positive"]) == 10
        assert len(paths["negative"]) == 10

    def test_paths_are_sorted(self, sample_dirs):
        pos_dir, neg_dir = sample_dirs
        ds = AlarmDataset(positive_dir=pos_dir, negative_dir=neg_dir)
        paths = ds.file_paths
        assert paths["positive"] == sorted(paths["positive"])
        assert paths["negative"] == sorted(paths["negative"])

    def test_paths_are_strings(self, sample_dirs):
        pos_dir, neg_dir = sample_dirs
        ds = AlarmDataset(positive_dir=pos_dir, negative_dir=neg_dir)
        paths = ds.file_paths
        for p in paths["positive"] + paths["negative"]:
            assert isinstance(p, str)


# ── TestBuildMetadata ────────────────────────────────────────────────────────

class TestBuildMetadata:
    def test_contains_all_top_level_keys(self, sample_dirs):
        pos_dir, neg_dir = sample_dirs
        ds = AlarmDataset(positive_dir=pos_dir, negative_dir=neg_dir)
        train_ds, val_ds = make_splits(ds, train_fraction=0.8)
        model = AlarmCNN()

        meta = _build_metadata(
            model_filename="alarm_model_2026-03-17.pt",
            created_date="2026-03-17",
            training_duration_seconds=42.5,
            n_epochs=10,
            best_epoch=7,
            target_accuracy=0.999,
            target_reached=True,
            full_dataset=ds,
            train_ds=train_ds,
            val_ds=val_ds,
            performance={
                "val_loss": 0.05, "val_accuracy": 0.98,
                "precision": 0.97, "recall": 0.99,
                "f1": 0.98, "false_positive_rate": 0.03,
                "false_negative_rate": 0.01,
            },
            lr=0.001,
            batch_size=16,
            model=model,
        )

        expected_keys = {
            "model_filename", "created_date", "training_duration_seconds",
            "epochs", "best_epoch", "target_accuracy_reached",
            "training_samples", "validation_samples", "model_performance",
            "training_config", "feature_config", "model_architecture",
            "environment", "notes",
        }
        assert set(meta.keys()) == expected_keys

    def test_training_samples_has_file_lists(self, sample_dirs):
        pos_dir, neg_dir = sample_dirs
        ds = AlarmDataset(positive_dir=pos_dir, negative_dir=neg_dir)
        train_ds, val_ds = make_splits(ds, train_fraction=0.8)
        model = AlarmCNN()

        meta = _build_metadata(
            model_filename="test.pt",
            created_date="2026-03-17",
            training_duration_seconds=1.0,
            n_epochs=1,
            best_epoch=1,
            target_accuracy=0.999,
            target_reached=False,
            full_dataset=ds,
            train_ds=train_ds,
            val_ds=val_ds,
            performance={
                "val_loss": 0.5, "val_accuracy": 0.5,
                "precision": 0.5, "recall": 0.5,
                "f1": 0.5, "false_positive_rate": 0.5,
                "false_negative_rate": 0.5,
            },
            lr=0.001,
            batch_size=16,
            model=model,
        )

        ts = meta["training_samples"]
        assert ts["total"] == 20
        assert ts["positive"] == 10
        assert ts["negative"] == 10
        assert len(ts["positive_files"]) == 10
        assert len(ts["negative_files"]) == 10

    def test_serialisable_to_json(self, sample_dirs):
        pos_dir, neg_dir = sample_dirs
        ds = AlarmDataset(positive_dir=pos_dir, negative_dir=neg_dir)
        train_ds, val_ds = make_splits(ds, train_fraction=0.8)
        model = AlarmCNN()

        meta = _build_metadata(
            model_filename="test.pt",
            created_date="2026-03-17",
            training_duration_seconds=1.0,
            n_epochs=1,
            best_epoch=1,
            target_accuracy=0.999,
            target_reached=False,
            full_dataset=ds,
            train_ds=train_ds,
            val_ds=val_ds,
            performance={
                "val_loss": 0.5, "val_accuracy": 0.5,
                "precision": 0.5, "recall": 0.5,
                "f1": 0.5, "false_positive_rate": 0.5,
                "false_negative_rate": 0.5,
            },
            lr=0.001,
            batch_size=16,
            model=model,
        )

        # Should not raise
        text = json.dumps(meta, indent=2)
        loaded = json.loads(text)
        assert loaded["model_filename"] == "test.pt"
