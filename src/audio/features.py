"""
audio/features.py
-----------------
Converts a raw 1-D audio window (numpy float32 array) into a log-mel-
spectrogram PyTorch tensor suitable for the AlarmCNN classifier.

The same transform is used identically during training *and* inference so the
model always sees the same feature representation.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio.transforms as T

from src.audio.capture import SAMPLE_RATE  # single source of truth

# ── Feature configuration ─────────────────────────────────────────────────────
# These must stay fixed once a model is trained – changing them invalidates
# any saved weights.
N_MELS: int = 64          # number of mel filter-bank bins
N_FFT: int = 1024         # FFT window length
HOP_LENGTH: int = 512     # STFT hop length
F_MIN: float = 50.0       # lowest mel frequency (Hz) – alarm tones rarely < 50 Hz
F_MAX: float = 8000.0     # highest mel frequency (Hz) – USB mic top-end

# Pre-build the transform once (thread-safe; stateless operation)
_mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX,
    power=2.0,   # power spectrogram
)
_amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)


def extract(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Convert a mono audio window to a log-mel-spectrogram tensor.

    Parameters
    ----------
    audio:
        1-D numpy float32 array, shape ``(n_samples,)``.
    sample_rate:
        Sample rate of the input audio.  Must match ``SAMPLE_RATE`` unless
        you know what you're doing.

    Returns
    -------
    torch.Tensor
        Shape ``(1, N_MELS, time_frames)`` – channel-first, suitable for
        feeding directly into a Conv2d layer.
    """
    if audio.ndim != 1:
        raise ValueError(f"Expected 1-D audio array, got shape {audio.shape}")

    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, n_samples)

    mel = _mel_transform(waveform)          # (1, N_MELS, time)
    log_mel = _amplitude_to_db(mel)         # (1, N_MELS, time)  in dB

    # Normalise to approximately [-1, 1] using the top_db range
    log_mel = log_mel / 40.0 - 1.0

    return log_mel  # (1, N_MELS, time_frames)
