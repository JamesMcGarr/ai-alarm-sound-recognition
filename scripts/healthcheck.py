#!/usr/bin/env python3
"""
scripts/healthcheck.py
----------------------
Health check script for ai-alarm-sound-recognition.

Run all checks:
    python scripts/healthcheck.py

Run specific checks:
    python scripts/healthcheck.py --env-vars --model-exists
    python scripts/healthcheck.py --tapo-on
    python scripts/healthcheck.py --tapo-off
    python scripts/healthcheck.py --tapo-connect

Available flags:
    --env-vars        Verify API_USERNAME, API_PASSWORD, DEVICE_IP_ADDRESS are set
    --imports         Verify all required Python packages are importable
    --model-exists    Verify the model file exists on disk
    --model-loads     Verify the model file loads correctly into AlarmCNN
    --audio-device    Verify the audio input device is accessible
    --tapo-connect    Verify credentials and connectivity to the Tapo plug
    --tapo-on         Turn the plug ON then restore it to OFF
    --tapo-off        Turn the plug OFF

    --model PATH      Override the model path (default: models/alarm_model.pt)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Callable

# ── path setup ────────────────────────────────────────────────────────────────
# Allow running from any directory without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv()

# ── ANSI colours ──────────────────────────────────────────────────────────────
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _green(text: str) -> str:
    return f"{_GREEN}{text}{_RESET}"


def _red(text: str) -> str:
    return f"{_RED}{text}{_RESET}"


def _yellow(text: str) -> str:
    return f"{_YELLOW}{text}{_RESET}"


def _bold(text: str) -> str:
    return f"{_BOLD}{text}{_RESET}"


# ── result tracking ───────────────────────────────────────────────────────────
_PASS = "PASS"
_FAIL = "FAIL"
_SKIP = "SKIP"

CheckResult = tuple[str, str]  # (status, detail)


def run_check(label: str, fn: Callable[[], CheckResult]) -> bool:
    """
    Run a single check function, print the result, and return True if it passed.
    """
    try:
        status, detail = fn()
    except Exception as exc:
        status, detail = _FAIL, str(exc)

    if status == _PASS:
        icon = _green(f"✓ {_PASS}")
    elif status == _SKIP:
        icon = _yellow(f"~ {_SKIP}")
    else:
        icon = _red(f"✗ {_FAIL}")

    suffix = f"  {detail}" if detail else ""
    print(f"  {icon}  {label}{suffix}")

    return status == _PASS


# ── individual checks ─────────────────────────────────────────────────────────

def check_env_vars() -> CheckResult:
    """Verify the three required Tapo environment variables are set."""
    required = ["API_USERNAME", "API_PASSWORD", "DEVICE_IP_ADDRESS"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        return _FAIL, f"Missing: {', '.join(missing)}"
    return _PASS, f"All set (device: {os.environ['DEVICE_IP_ADDRESS']})"


def check_imports() -> CheckResult:
    """Verify all required Python packages can be imported."""
    packages = [
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("sounddevice", "sounddevice"),
        ("tapo", "tapo"),
        ("dotenv", "python-dotenv"),
        ("scipy", "scipy"),
        ("numpy", "numpy"),
    ]
    failed: list[str] = []
    import importlib

    for module, pkg_name in packages:
        try:
            importlib.import_module(module)
        except ImportError:
            failed.append(pkg_name)

    if failed:
        return _FAIL, f"Not importable: {', '.join(failed)}"
    return _PASS, f"{len(packages)} packages OK"


def check_model_exists(model_path: Path) -> CheckResult:
    """Verify the model file exists on disk."""
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        return _PASS, f"{model_path}  ({size_mb:.1f} MB)"
    return _FAIL, f"Not found: {model_path}"


def check_model_loads(model_path: Path) -> CheckResult:
    """Load the model file and verify it returns an AlarmCNN instance."""
    from src.training.trainer import load_model
    from src.training.model import AlarmCNN

    if not model_path.exists():
        return _FAIL, f"Model file not found: {model_path}"

    model = load_model(model_path)
    if not isinstance(model, AlarmCNN):
        return _FAIL, f"Expected AlarmCNN, got {type(model).__name__}"
    return _PASS, f"AlarmCNN loaded ({sum(p.numel() for p in model.parameters()):,} params)"


def check_audio_device() -> CheckResult:
    """Verify the audio input device is accessible via sounddevice."""
    try:
        import sounddevice as sd
    except OSError as exc:
        return _FAIL, f"PortAudio unavailable: {exc}"

    from src.audio.capture import DEVICE_INDEX, SAMPLE_RATE

    device_index = int(os.environ.get("AUDIO_DEVICE", DEVICE_INDEX))
    configured_rate = SAMPLE_RATE
    try:
        info = sd.query_devices(device_index, kind="input")
    except Exception as exc:
        return _FAIL, f"Device {device_index} not found: {exc}"

    name = info.get("name", "?")
    device_rate = int(info.get("default_samplerate", 0))
    mismatch = f"  ⚠ device reports {device_rate} Hz but SAMPLE_RATE={configured_rate}" if device_rate != configured_rate else ""
    return _PASS, f"Device {device_index}: '{name}'  (SAMPLE_RATE={configured_rate} Hz){mismatch}"


def _tapo_creds_present() -> bool:
    return all(os.environ.get(v) for v in ["API_USERNAME", "API_PASSWORD", "DEVICE_IP_ADDRESS"])


def check_tapo_connect() -> CheckResult:
    """Connect to the Tapo plug and verify credentials + network reachability."""
    if not _tapo_creds_present():
        return _SKIP, "Credentials not set — run --env-vars first"

    from src.siren.tapo_client import TapoClient, TapoDevice

    client = TapoClient()
    ip = os.environ["DEVICE_IP_ADDRESS"]
    device = TapoDevice(client.api_client, ip, "healthcheck")
    asyncio.run(device.initialize())
    return _PASS, f"Connected to {ip}"


def check_tapo_on() -> CheckResult:
    """
    Turn the Tapo plug ON, then restore it to OFF regardless of outcome.
    """
    if not _tapo_creds_present():
        return _SKIP, "Credentials not set — run --env-vars first"

    from src.siren.controller import SirenController

    controller = SirenController()
    try:
        controller.turn_on()
    finally:
        # Always restore to off, even if turn_on raised.
        try:
            controller._siren_on = None  # force the off command to be sent
            controller.turn_off()
        except Exception:
            pass

    return _PASS, f"Turned ON then restored to OFF  (device: {os.environ['DEVICE_IP_ADDRESS']})"


def check_tapo_off() -> CheckResult:
    """Turn the Tapo plug OFF."""
    if not _tapo_creds_present():
        return _SKIP, "Credentials not set — run --env-vars first"

    from src.siren.controller import SirenController

    controller = SirenController()
    controller.turn_off()
    return _PASS, f"Turned OFF  (device: {os.environ['DEVICE_IP_ADDRESS']})"


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="healthcheck",
        description=(
            "Health check for ai-alarm-sound-recognition. "
            "Runs all checks when called with no arguments."
        ),
    )
    parser.add_argument(
        "--model",
        metavar="PATH",
        default=None,
        help="Path to the model file (default: models/alarm_model.pt)",
    )
    parser.add_argument("--env-vars", action="store_true", help="Check required env vars")
    parser.add_argument("--imports", action="store_true", help="Check Python package imports")
    parser.add_argument("--model-exists", action="store_true", help="Check model file exists")
    parser.add_argument("--model-loads", action="store_true", help="Check model file loads")
    parser.add_argument("--audio-device", action="store_true", help="Check audio input device")
    parser.add_argument("--tapo-connect", action="store_true", help="Check Tapo plug connectivity")
    parser.add_argument("--tapo-on", action="store_true", help="Turn Tapo plug ON (then restore OFF)")
    parser.add_argument("--tapo-off", action="store_true", help="Turn Tapo plug OFF")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve model path without importing src.training.trainer — that module
    # chains through src.audio.capture which queries the audio device at import
    # time and will crash if the device is absent.
    model_path = Path(args.model) if args.model else _ROOT / "models" / "alarm_model.pt"

    # Determine which checks to run
    explicit_flags = [
        args.env_vars,
        args.imports,
        args.model_exists,
        args.model_loads,
        args.audio_device,
        args.tapo_connect,
        args.tapo_on,
        args.tapo_off,
    ]
    run_all = not any(explicit_flags)

    # Build the ordered check list: (label, flag_active, fn)
    checks: list[tuple[str, bool, Callable[[], CheckResult]]] = [
        ("Environment variables",   run_all or args.env_vars,      check_env_vars),
        ("Python imports",          run_all or args.imports,        check_imports),
        ("Model file exists",       run_all or args.model_exists,   lambda: check_model_exists(model_path)),
        ("Model loads",             run_all or args.model_loads,    lambda: check_model_loads(model_path)),
        ("Audio device",            run_all or args.audio_device,   check_audio_device),
        ("Tapo connect",            run_all or args.tapo_connect,   check_tapo_connect),
        ("Tapo turn ON",            run_all or args.tapo_on,        check_tapo_on),
        ("Tapo turn OFF",           run_all or args.tapo_off,       check_tapo_off),
    ]

    print()
    print(_bold("ai-alarm-sound-recognition — health check"))
    print("─" * 50)

    passed = 0
    total = 0

    for label, active, fn in checks:
        if not active:
            continue
        total += 1
        if run_check(label, fn):
            passed += 1

    print("─" * 50)
    if passed == total:
        summary = _green(f"{passed}/{total} checks passed")
    else:
        failed = total - passed
        summary = _red(f"{passed}/{total} checks passed — {failed} failed")

    print(f"  {summary}")
    print()

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
