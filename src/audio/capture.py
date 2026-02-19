"""
audio/capture.py
----------------
Streams overlapping fixed-length windows from a USB microphone using sounddevice.

The device name/index can be overridden via AUDIO_DEVICE env var or the
`device` argument.  Defaults match the ALSA device used in record_sample.sh
(plughw:2,0 → typically device index 2 in sounddevice).
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Generator
from typing import Optional

import numpy as np
import sounddevice as sd

# ── defaults ──────────────────────────────────────────────────────────────────
DEVICE_INDEX: int = 0          # USB PnP Sound Device (hw:2,0) – index 0 in sounddevice


def get_sample_rate(device_index: int | None = None) -> int:
    """Query the device's native sample rate rather than hardcoding one."""
    info = sd.query_devices(device_index, kind="input")
    return int(info["default_samplerate"])


SAMPLE_RATE: int = get_sample_rate(DEVICE_INDEX)
WINDOW_SECONDS: float = 1.0    # duration of each audio window
HOP_SECONDS: float = 0.5       # overlap step between windows
CHANNELS: int = 1


class AudioCapture:
    """
    Continuously records from a microphone and exposes non-overlapping or
    overlapping windows as a blocking generator.

    Parameters
    ----------
    device:
        sounddevice device index or substring of its name.
        ``None`` → system default (override with AUDIO_DEVICE env var).
    sample_rate:
        Samples per second.
    window_seconds:
        Duration of each yielded window in seconds.
    hop_seconds:
        Step between successive window start positions.  Smaller values →
        more overlap → more windows per second.  Must be ≤ window_seconds.
    """

    def __init__(
        self,
        device: Optional[int | str] = DEVICE_INDEX,
        sample_rate: int = SAMPLE_RATE,
        window_seconds: float = WINDOW_SECONDS,
        hop_seconds: float = HOP_SECONDS,
    ) -> None:
        import os

        self.device = device or os.environ.get("AUDIO_DEVICE")
        self.sample_rate = sample_rate
        self.window_size = int(window_seconds * sample_rate)
        self.hop_size = int(hop_seconds * sample_rate)
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._ring: np.ndarray = np.zeros(self.window_size, dtype=np.float32)
        self._ring_pos: int = 0
        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None

    # ── internal callback ─────────────────────────────────────────────────────

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice on every audio block (runs in audio thread)."""
        if status:
            return  # silently skip glitched frames

        chunk = indata[:, 0].copy()  # mono
        written = 0

        while written < len(chunk):
            space = self.window_size - self._ring_pos
            take = min(space, len(chunk) - written)
            self._ring[self._ring_pos : self._ring_pos + take] = chunk[
                written : written + take
            ]
            self._ring_pos += take
            written += take

            if self._ring_pos >= self.window_size:
                # full window ready → push a copy and shift by hop_size
                self._q.put(self._ring.copy())
                self._ring[: self.window_size - self.hop_size] = self._ring[
                    self.hop_size :
                ]
                self._ring_pos = self.window_size - self.hop_size

    # ── public API ────────────────────────────────────────────────────────────

    def stream(self, timeout: Optional[float] = None) -> Generator[np.ndarray, None, None]:
        """
        Yields numpy arrays of shape ``(window_size,)`` dtype float32,
        normalised to [-1, 1].  Blocks until each window is ready.

        Parameters
        ----------
        timeout:
            Seconds to wait for each window before raising ``StopIteration``.
            ``None`` → block indefinitely.

        Yields
        ------
        np.ndarray
            Audio window, shape ``(window_size,)``, dtype float32.
        """
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        with self._stream:
            while True:
                try:
                    yield self._q.get(timeout=timeout)
                except queue.Empty:
                    return

    def record_clip(self, duration_seconds: float) -> np.ndarray:
        """
        Blocking convenience method: records exactly ``duration_seconds`` of
        audio and returns the concatenated float32 array.

        Parameters
        ----------
        duration_seconds:
            How many seconds to record.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples,)``, dtype float32.
        """
        n_samples = int(duration_seconds * self.sample_rate)
        recording = sd.rec(
            n_samples,
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype="float32",
            device=self.device,
        )
        sd.wait()
        return recording[:, 0]
