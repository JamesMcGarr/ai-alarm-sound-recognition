"""
tests/test_model.py
-------------------
Unit tests for AlarmCNN.
"""

import torch
import pytest

from src.audio.features import N_MELS
from src.training.model import AlarmCNN, DETECTION_THRESHOLD


def _dummy_batch(batch_size: int = 4, time_frames: int = 44) -> torch.Tensor:
    """Return random spectrogram batch of shape (B, 1, N_MELS, T)."""
    return torch.randn(batch_size, 1, N_MELS, time_frames)


class TestAlarmCNN:
    def test_forward_output_shape(self):
        model = AlarmCNN()
        x = _dummy_batch()
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_in_0_1(self):
        model = AlarmCNN()
        x = _dummy_batch()
        out = model(x)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_predict_proba_single(self):
        model = AlarmCNN()
        x = _dummy_batch(batch_size=1)
        prob = model.predict_proba(x)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_predict_proba_auto_adds_batch_dim(self):
        """predict_proba should accept a (1, N_MELS, T) tensor without batch dim."""
        model = AlarmCNN()
        x = torch.randn(1, N_MELS, 44)
        prob = model.predict_proba(x)
        assert isinstance(prob, float)

    def test_is_alarm_returns_bool(self):
        model = AlarmCNN()
        x = _dummy_batch(batch_size=1)
        result = model.is_alarm(x)
        assert isinstance(result, bool)

    def test_eval_mode_in_predict(self):
        model = AlarmCNN()
        model.train()
        _ = model.predict_proba(_dummy_batch(batch_size=1))
        assert not model.training, "predict_proba must put the model into eval mode"

    def test_parameter_count_reasonable(self):
        """Model should be compact – under 1 M parameters."""
        model = AlarmCNN()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 1_000_000, f"Model has {n_params} params – unexpectedly large"
