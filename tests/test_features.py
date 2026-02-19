"""
tests/test_features.py
----------------------
Unit tests for the mel-spectrogram feature extractor.
"""

import numpy as np
import pytest
import torch

from src.audio.features import extract, N_MELS, SAMPLE_RATE


def _sine_wave(freq: float = 1000.0, duration: float = 1.0) -> np.ndarray:
    """Generate a normalised sine wave as float32."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class TestExtract:
    def test_output_shape(self):
        audio = _sine_wave()
        tensor = extract(audio)
        assert tensor.ndim == 3, "Output must be 3-D (C, N_MELS, T)"
        assert tensor.shape[0] == 1, "Channel dim must be 1"
        assert tensor.shape[1] == N_MELS, f"Mel bins must be {N_MELS}"

    def test_output_dtype(self):
        audio = _sine_wave()
        tensor = extract(audio)
        assert tensor.dtype == torch.float32

    def test_output_is_finite(self):
        audio = _sine_wave()
        tensor = extract(audio)
        assert torch.isfinite(tensor).all(), "Spectrogram must not contain NaN or Inf"

    def test_silent_input(self):
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        tensor = extract(audio)
        # Silent input should still produce a valid tensor, not crash
        assert tensor.shape[1] == N_MELS

    def test_rejects_2d_input(self):
        audio = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            extract(audio)

    def test_different_frequencies_differ(self):
        """Spectrograms of different tones should not be identical."""
        t1 = extract(_sine_wave(freq=500.0))
        t2 = extract(_sine_wave(freq=4000.0))
        assert not torch.allclose(t1, t2), "Different tones must produce different spectrograms"
