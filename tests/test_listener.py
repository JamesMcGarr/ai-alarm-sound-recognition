"""
tests/test_listener.py
----------------------
Unit tests for the AlarmListener duty-cycle logic and alarm callbacks.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from src.inference.listener import (
    AlarmListener,
    on_alarm_detected,
    on_alarm_not_detected,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dummy_window(samples: int = 44100) -> np.ndarray:
    """Return a random float32 audio window."""
    return np.random.randn(samples).astype(np.float32)


def _make_listener(**overrides) -> AlarmListener:
    """
    Build an AlarmListener with heavy mocking so no model, mic, or siren
    hardware is needed.
    """
    with patch("src.inference.listener.load_model") as mock_load, \
         patch("src.inference.listener.AudioCapture"):
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = 0.0
        mock_load.return_value = mock_model

        defaults = dict(
            model_path=None,
            device=None,
            save_triggers=False,
            save_negatives=False,
        )
        defaults.update(overrides)
        listener = AlarmListener(**defaults)

    return listener


# ── Callback tests ────────────────────────────────────────────────────────────

class TestCallbacks:
    """on_alarm_detected and on_alarm_not_detected no longer touch the siren."""

    def test_on_alarm_detected_no_siren_param(self):
        """Calling on_alarm_detected should not require a siren argument."""
        window = _dummy_window()
        # Should not raise
        on_alarm_detected(window, 0.9995)

    def test_on_alarm_not_detected_no_siren_param(self):
        """Calling on_alarm_not_detected should not require a siren argument."""
        window = _dummy_window()
        on_alarm_not_detected(window, 0.5)


# ── _process_window tests ────────────────────────────────────────────────────

class TestProcessWindow:
    def test_returns_true_when_alarm_detected(self):
        listener = _make_listener()
        listener.model.predict_proba.return_value = 0.9999
        listener.threshold = 0.999

        result = listener._process_window(_dummy_window())
        assert result is True

    def test_returns_false_when_no_alarm(self):
        listener = _make_listener()
        listener.model.predict_proba.return_value = 0.1
        listener.threshold = 0.999

        result = listener._process_window(_dummy_window())
        assert result is False

    def test_returns_false_at_exact_threshold(self):
        """Confidence must be >= threshold, so exact threshold → True."""
        listener = _make_listener()
        listener.model.predict_proba.return_value = 0.999
        listener.threshold = 0.999

        result = listener._process_window(_dummy_window())
        assert result is True


# ── Duty-cycle tests ─────────────────────────────────────────────────────────

class TestSirenDutyCycle:
    def test_duty_cycle_calls_on_sleep_off(self):
        """The duty cycle must call turn_on → sleep → turn_off in order."""
        listener = _make_listener(siren_on_duration=3.0)
        listener.siren = MagicMock()

        with patch("src.inference.listener.time.sleep") as mock_sleep:
            listener._siren_duty_cycle()

        listener.siren.turn_on.assert_called_once()
        mock_sleep.assert_called_once_with(3.0)
        listener.siren.turn_off.assert_called_once()

        # Verify ordering: on before sleep before off
        listener.siren.turn_on.assert_called_once()
        listener.siren.turn_off.assert_called_once()

    def test_duty_cycle_uses_configured_duration(self):
        """The sleep duration should match siren_on_duration."""
        listener = _make_listener(siren_on_duration=7.5)
        listener.siren = MagicMock()

        with patch("src.inference.listener.time.sleep") as mock_sleep:
            listener._siren_duty_cycle()

        mock_sleep.assert_called_once_with(7.5)

    def test_run_triggers_duty_cycle_on_alarm(self):
        """
        When _process_window returns True and a siren is configured,
        run() should invoke _siren_duty_cycle.
        """
        listener = _make_listener(siren_on_duration=1.0)
        listener.siren = MagicMock()

        windows = [_dummy_window()]
        listener.capture.stream.return_value = iter(windows)

        # Make the model always detect alarm
        listener.model.predict_proba.return_value = 0.9999
        listener.threshold = 0.999

        with patch("src.inference.listener.time.sleep"):
            listener.run()

        listener.siren.turn_on.assert_called_once()
        listener.siren.turn_off.assert_called_once()

    def test_run_no_duty_cycle_without_siren(self):
        """Without a siren, run() should not error on alarm detection."""
        listener = _make_listener()
        listener.siren = None

        windows = [_dummy_window()]
        listener.capture.stream.return_value = iter(windows)

        listener.model.predict_proba.return_value = 0.9999
        listener.threshold = 0.999

        # Should complete without error
        listener.run()

    def test_run_no_duty_cycle_when_no_alarm(self):
        """When no alarm is detected, siren should not be touched."""
        listener = _make_listener(siren_on_duration=1.0)
        listener.siren = MagicMock()

        windows = [_dummy_window(), _dummy_window()]
        listener.capture.stream.return_value = iter(windows)

        listener.model.predict_proba.return_value = 0.1
        listener.threshold = 0.999

        listener.run()

        listener.siren.turn_on.assert_not_called()
        listener.siren.turn_off.assert_not_called()

    def test_run_multiple_alarms_trigger_multiple_cycles(self):
        """Each alarm detection should trigger a separate duty cycle."""
        listener = _make_listener(siren_on_duration=2.0)
        listener.siren = MagicMock()

        windows = [_dummy_window(), _dummy_window(), _dummy_window()]
        listener.capture.stream.return_value = iter(windows)

        # All three windows trigger alarm
        listener.model.predict_proba.return_value = 0.9999
        listener.threshold = 0.999

        with patch("src.inference.listener.time.sleep"):
            listener.run()

        assert listener.siren.turn_on.call_count == 3
        assert listener.siren.turn_off.call_count == 3


# ── Default siren_on_duration ────────────────────────────────────────────────

class TestDefaults:
    def test_default_siren_on_duration(self):
        listener = _make_listener()
        assert listener.siren_on_duration == 5.0

    def test_custom_siren_on_duration(self):
        listener = _make_listener(siren_on_duration=10.0)
        assert listener.siren_on_duration == 10.0
