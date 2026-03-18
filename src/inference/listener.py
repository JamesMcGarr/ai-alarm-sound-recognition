"""
inference/listener.py
---------------------
Continuously listens to the microphone and fires ``on_alarm_detected`` when
the trained AlarmCNN model scores a window at ≥ 99.9% confidence.

Nice-to-haves implemented:
  • Every alarm trigger is saved to data/positive_captures/ as a timestamped
    .wav file for future retraining.
  • "Interesting" non-alarm sounds (non-silent frames that score < the
    detection threshold) are saved to data/negative_captures/ as hard
    negatives for retraining.
"""

from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import scipy.io.wavfile as wavfile

from src.audio.capture import AudioCapture, SAMPLE_RATE
from src.audio.features import extract
from src.training.model import AlarmCNN, DETECTION_THRESHOLD
from src.training.trainer import load_model

if TYPE_CHECKING:
    from src.siren.controller import SirenController

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
POSITIVE_CAPTURES_DIR = DATA_DIR / "positive_captures"
NEGATIVE_CAPTURES_DIR = DATA_DIR / "negative_captures"

# RMS energy threshold below which a window is considered "silent" and not
# worth saving as a negative sample.  Tune if your environment is noisy.
SILENCE_RMS_THRESHOLD: float = 0.005

# Minimum model confidence required to save a frame as an interesting negative
# (avoids saving pure silence; set to 0.0 to save all non-silent non-alarms)
INTERESTING_MIN_CONFIDENCE: float = 0.1


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _save_wav(audio: np.ndarray, path: Path) -> None:
    """Save a float32 numpy array as a 16-bit .wav file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    wavfile.write(str(path), SAMPLE_RATE, pcm)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


# ── Alarm callbacks ───────────────────────────────────────────────────────────

def on_alarm_detected(
    audio_window: np.ndarray,
    confidence: float,
) -> None:
    """
    Called every time the model detects the X-Sense alarm sound.

    Handles logging only.  Siren control is managed by the duty-cycle
    logic in :meth:`AlarmListener.run`.

    Parameters
    ----------
    audio_window:
        The raw audio frame that triggered the detection.
    confidence:
        The model's output probability (≥ DETECTION_THRESHOLD).
    """
    logger.info("🔔  ALARM DETECTED  (confidence=%.4f)", confidence)


def on_alarm_not_detected(
    audio_window: np.ndarray,
    confidence: float,
) -> None:
    """
    Called for every audio window that does NOT trigger an alarm detection.

    Handles logging only.  Siren control is managed by the duty-cycle
    logic in :meth:`AlarmListener.run`.

    Parameters
    ----------
    audio_window:
        The raw audio frame that was analysed.
    confidence:
        The model's output probability (< DETECTION_THRESHOLD).
    """
    pass


# ── Main listener ─────────────────────────────────────────────────────────────

class AlarmListener:
    """
    Loads the trained model and continuously analyses live audio frames.

    Parameters
    ----------
    model_path:
        Path to the saved ``.pt`` weights file.
    device:
        sounddevice device index or name fragment for the USB microphone.
    detection_threshold:
        Minimum model confidence to fire ``on_alarm_detected``.
    save_triggers:
        Whether to save triggering frames to ``data/positive_captures/``.
    save_negatives:
        Whether to save interesting non-triggering frames to
        ``data/negative_captures/``.
    siren_on_duration:
        Seconds the siren stays on per duty-cycle pulse.  The siren is
        turned on when the alarm is first detected, held for this many
        seconds, then turned off.  Normal listening resumes automatically;
        if the alarm is still audible the cycle repeats.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[int | str] = None,
        detection_threshold: float = DETECTION_THRESHOLD,
        save_triggers: bool = True,
        save_negatives: bool = True,
        siren: "SirenController | None" = None,
        siren_on_duration: float = 5.0,
    ) -> None:
        self.model = load_model(model_path) if model_path else load_model()
        self.capture = AudioCapture(device=device)
        self.threshold = detection_threshold
        self.save_triggers = save_triggers
        self.save_negatives = save_negatives
        self.siren = siren
        self.siren_on_duration = siren_on_duration

        if save_triggers:
            POSITIVE_CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        if save_negatives:
            NEGATIVE_CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """
        Block and listen indefinitely.  Press Ctrl+C to stop.

        For every audio window:
        1. Extract mel-spectrogram features.
        2. Run the model.
        3. If confidence ≥ threshold → call ``on_alarm_detected`` + optionally save.
        4. Otherwise, if the window has interesting energy → optionally save as negative.

        When the siren is configured the listener uses a **duty-cycle**
        approach to avoid the siren's own noise preventing detection of
        the X-Sense alarm:

        1. Alarm detected → turn siren ON.
        2. Hold siren ON for ``siren_on_duration`` seconds.
        3. Turn siren OFF and resume normal listening.
        4. If the alarm is detected again the cycle repeats.

        Audio captured while the siren is blaring will not match the
        X-Sense alarm signature, so no explicit gap or queue-draining
        is needed — the model naturally ignores it.
        """
        logger.info("Listening…  (Ctrl+C to stop)")
        if self.siren is not None:
            logger.info(
                "Siren duty-cycle enabled: %.1f s ON per pulse.",
                self.siren_on_duration,
            )
        try:
            for window in self.capture.stream():
                alarm_detected = self._process_window(window)
                if alarm_detected and self.siren is not None:
                    self._siren_duty_cycle()
        except KeyboardInterrupt:
            logger.info("Listener stopped.")

    def _siren_duty_cycle(self) -> None:
        """Run one siren duty-cycle pulse: ON → hold → OFF."""
        logger.info(
            "⏱  Siren duty-cycle: ON for %.1f s", self.siren_on_duration
        )
        self.siren.turn_on()
        time.sleep(self.siren_on_duration)
        self.siren.turn_off()
        logger.info("⏱  Siren duty-cycle: OFF — resuming listening.")

    def _process_window(self, window: np.ndarray) -> bool:
        """Analyse one audio window.  Returns ``True`` if alarm detected."""
        spec = extract(window)
        confidence = self.model.predict_proba(spec)

        if confidence >= self.threshold:
            on_alarm_detected(window, confidence)
            if self.save_triggers:
                path = POSITIVE_CAPTURES_DIR / f"trigger_{_timestamp()}.wav"
                _save_wav(window, path)
                logger.debug("Saved trigger → %s", path.name)
            return True
        else:
            on_alarm_not_detected(window, confidence)
            rms = _rms(window)
            if (
                self.save_negatives
                and rms >= SILENCE_RMS_THRESHOLD
                and confidence >= INTERESTING_MIN_CONFIDENCE
            ):
                path = NEGATIVE_CAPTURES_DIR / f"negative_{_timestamp()}.wav"
                _save_wav(window, path)
                logger.debug(
                    "Saved interesting negative → %s  (conf=%.3f, rms=%.4f)",
                    path.name, confidence, rms
                )
            return False
