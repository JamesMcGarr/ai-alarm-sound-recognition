"""
listen.py
---------
Entrypoint for live inference.  Loads the trained model and continuously
analyses audio from the USB microphone, calling on_alarm_detected() whenever
the X-Sense alarm is recognised with ≥ 99% confidence.

Usage
-----
    python listen.py

Options
-------
--device INDEX|NAME   sounddevice device index or name fragment.
--threshold FLOAT     Detection confidence threshold (default: 0.99).
--no-save-triggers    Do not save triggering frames to data/positive_captures/.
--no-save-negatives   Do not save interesting non-alarm frames to data/negative_captures/.
--model PATH          Override path to the trained .pt model file.
--verbose             Enable DEBUG logging (shows every saved capture).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live alarm detection using the trained AlarmCNN model."
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="INDEX|NAME",
        help="sounddevice device index or name fragment for the USB microphone.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        metavar="FLOAT",
        help="Detection confidence threshold (default: 0.99).",
    )
    p.add_argument(
        "--no-save-triggers",
        dest="save_triggers",
        action="store_false",
        default=True,
        help="Disable saving alarm-trigger frames to data/positive_captures/.",
    )
    p.add_argument(
        "--no-save-negatives",
        dest="save_negatives",
        action="store_false",
        default=True,
        help="Disable saving interesting non-alarm frames to data/negative_captures/.",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a trained alarm_model.pt file (default: models/alarm_model.pt).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.inference.listener import AlarmListener
    from src.training.trainer import MODEL_PATH

    model_path = args.model or MODEL_PATH
    if not model_path.exists():
        logging.error(
            "Model file not found: %s\n"
            "Train the model first with:  python train.py",
            model_path,
        )
        return 1

    device = args.device
    if device is not None:
        try:
            device = int(device)
        except ValueError:
            pass  # keep as string name

    listener = AlarmListener(
        model_path=model_path,
        device=device,
        detection_threshold=args.threshold,
        save_triggers=args.save_triggers,
        save_negatives=args.save_negatives,
    )

    logging.info("Model loaded from %s", model_path)
    logging.info("Detection threshold: %.2f", args.threshold)
    listener.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
