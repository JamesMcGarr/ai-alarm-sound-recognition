"""
training/collector.py
---------------------
Interactive CLI for recording labelled audio clips directly from the USB
microphone and saving them as .wav files to be used for training.

Layout:
    data/positive/   – alarm sound clips   (label = 1)
    data/negative/   – non-alarm clips     (label = 0)

Usage (from project root):
    python -m src.training.collector
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

from src.audio.capture import AudioCapture, SAMPLE_RATE

logger = logging.getLogger(__name__)

# ── directories ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
POSITIVE_DIR = DATA_DIR / "positive"
NEGATIVE_DIR = DATA_DIR / "negative"

CLIP_DURATION: float = 1.0          # seconds per labelled clip
DEFAULT_POSITIVE: int = 60          # recommended positive sample count
DEFAULT_NEGATIVE: int = 60          # recommended negative sample count


def _save_wav(audio: np.ndarray, path: Path) -> None:
    """Save a float32 array as a 16-bit PCM .wav file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    wavfile.write(str(path), SAMPLE_RATE, pcm)


def _next_filename(directory: Path, prefix: str) -> Path:
    """Return the next available sequential filename in *directory*."""
    existing = sorted(directory.glob(f"{prefix}_*.wav"))
    idx = len(existing)
    return directory / f"{prefix}_{idx:04d}.wav"


def _prompt_continue(message: str) -> None:
    input(f"\n{message}  [Press ENTER when ready] ")


def collect_samples(
    n_positive: int = DEFAULT_POSITIVE,
    n_negative: int = DEFAULT_NEGATIVE,
    device: int | str | None = None,
) -> None:
    """
    Interactively record *n_positive* alarm clips and *n_negative* background
    clips.  Each clip is CLIP_DURATION seconds long.

    Parameters
    ----------
    n_positive:
        Number of positive (alarm) samples to record.
    n_negative:
        Number of negative (background) samples to record.
    device:
        sounddevice device index or name fragment.  ``None`` → default.
    """
    capture = AudioCapture(device=device, sample_rate=SAMPLE_RATE)

    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)

    already_positive = len(list(POSITIVE_DIR.glob("positive_*.wav")))
    already_negative = len(list(NEGATIVE_DIR.glob("negative_*.wav")))

    if already_positive:
        logger.info("Found %d existing positive samples.", already_positive)
    if already_negative:
        logger.info("Found %d existing negative samples.", already_negative)

    # ── Positive samples ──────────────────────────────────────────────────────
    remaining_positive = n_positive
    if remaining_positive > 0:
        _prompt_continue(
            f"POSITIVE samples ({remaining_positive} clips × {CLIP_DURATION}s).\n"
            "  Trigger your X-Sense alarm so it is actively sounding."
        )
        logger.info("Recording %d positive clips …", remaining_positive)
        for i in range(remaining_positive):
            path = _next_filename(POSITIVE_DIR, "positive")
            audio = capture.record_clip(CLIP_DURATION)
            _save_wav(audio, path)
            logger.debug("  [%d/%d]  saved → %s", i + 1, remaining_positive, path.name)
        logger.info("Done. %d positive clips saved to %s", remaining_positive, POSITIVE_DIR)

    # ── Negative samples ──────────────────────────────────────────────────────
    remaining_negative = n_negative
    if remaining_negative > 0:
        _prompt_continue(
            f"NEGATIVE samples ({remaining_negative} clips × {CLIP_DURATION}s).\n"
            "  Let the alarm stop.  Make normal ambient background sounds\n"
            "  (speech, footsteps, TV, silence, etc.) during recording."
        )
        logger.info("Recording %d negative clips …", remaining_negative)
        for i in range(remaining_negative):
            path = _next_filename(NEGATIVE_DIR, "negative")
            audio = capture.record_clip(CLIP_DURATION)
            _save_wav(audio, path)
            logger.debug("  [%d/%d]  saved → %s", i + 1, remaining_negative, path.name)
        logger.info("Done. %d negative clips saved to %s", remaining_negative, NEGATIVE_DIR)

    total_pos = len(list(POSITIVE_DIR.glob("positive_*.wav")))
    total_neg = len(list(NEGATIVE_DIR.glob("negative_*.wav")))
    logger.info(
        "Collection complete. Positive samples: %d  |  Negative samples: %d",
        total_pos, total_neg,
    )


if __name__ == "__main__":
    collect_samples()
