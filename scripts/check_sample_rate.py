#!/usr/bin/env python3
"""
scripts/check_sample_rate.py
-----------------------------
Queries a sounddevice audio input device and reports its native sample rate.

Use this when switching to a different microphone to find out what value to
set in your .env file:

    python scripts/check_sample_rate.py
    python scripts/check_sample_rate.py --device 1

Then add or update this line in your .env:

    SAMPLE_RATE=<reported value>

If SAMPLE_RATE is not set in .env, the application defaults to 44100 Hz.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from any directory without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report the native sample rate of an audio input device."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="INDEX",
        help="sounddevice device index to query (default: DEVICE_INDEX from capture.py = 0)",
    )
    args = parser.parse_args(argv)

    try:
        import sounddevice as sd
    except OSError as exc:
        print(f"ERROR: PortAudio unavailable — {exc}", file=sys.stderr)
        return 1

    from src.audio.capture import DEVICE_INDEX, DEFAULT_SAMPLE_RATE

    device_index = args.device if args.device is not None else DEVICE_INDEX

    try:
        info = sd.query_devices(device_index, kind="input")
    except Exception as exc:
        print(f"ERROR: Could not query device {device_index} — {exc}", file=sys.stderr)
        print("\nAvailable input devices:", file=sys.stderr)
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                print(f"  [{i}] {dev['name']}", file=sys.stderr)
        return 1

    device_name = info["name"]
    device_rate = int(info["default_samplerate"])

    print(f"\nDevice [{device_index}]: {device_name}")
    print(f"Native sample rate: {device_rate} Hz")

    configured_rate = int(os.environ.get("SAMPLE_RATE", DEFAULT_SAMPLE_RATE))
    if os.environ.get("SAMPLE_RATE"):
        if configured_rate == device_rate:
            print(f"\nSAMPLE_RATE={configured_rate} in .env — matches device. ✓")
        else:
            print(f"\n⚠ SAMPLE_RATE={configured_rate} in .env does NOT match device ({device_rate} Hz).")
            print(f"  Update your .env:\n\n    SAMPLE_RATE={device_rate}\n")
    else:
        if device_rate == DEFAULT_SAMPLE_RATE:
            print(f"\nNo SAMPLE_RATE set in .env — default of {DEFAULT_SAMPLE_RATE} Hz matches device. ✓")
        else:
            print(f"\nNo SAMPLE_RATE set in .env — default is {DEFAULT_SAMPLE_RATE} Hz but device reports {device_rate} Hz.")
            print(f"  Add this to your .env:\n\n    SAMPLE_RATE={device_rate}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
