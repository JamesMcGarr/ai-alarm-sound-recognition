"""
train.py
--------
Entrypoint for data collection and model training.

Usage
-----
Collect samples then train (full workflow):
    python train.py

Collect only (no training):
    python train.py --collect

Train only (samples already collected):
    python train.py --train

Flags
-----
--collect           Run the interactive sample collection step.
--train             Run the training step.
--no-collect        Skip collection (same as --train alone).
--epochs N          Override MAX_EPOCHS (default: 200).
--positive N        Number of positive clips to record (default: 60).
--negative N        Number of negative clips to record (default: 60).
--device INDEX|NAME sounddevice device index or name for the USB mic.
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect alarm samples and train the AlarmCNN classifier."
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--collect",
        action="store_true",
        default=False,
        help="Run sample collection only (skip training).",
    )
    mode.add_argument(
        "--train",
        dest="train_only",
        action="store_true",
        default=False,
        help="Run training only (skip collection).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="Maximum training epochs (default: 200).",
    )
    p.add_argument(
        "--positive",
        type=int,
        default=60,
        metavar="N",
        help="Number of positive (alarm) clips to record (default: 60).",
    )
    p.add_argument(
        "--negative",
        type=int,
        default=60,
        metavar="N",
        help="Number of negative (background) clips to record (default: 60).",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="INDEX|NAME",
        help="sounddevice device index or name fragment for the USB microphone.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging("logs/train.log")
    args = parse_args(argv)

    do_collect = not args.train_only   # collect unless --train flag given
    do_train = not args.collect        # train unless --collect flag given

    if do_collect:
        from src.training.collector import collect_samples

        logger.info("=== Sample Collection ===")
        collect_samples(
            n_positive=args.positive,
            n_negative=args.negative,
            device=args.device if args.device is None else _parse_device(args.device),
        )

    if do_train:
        from src.training.trainer import train

        logger.info("=== Model Training ===")
        _model, best_acc = train(n_epochs=args.epochs)
        logger.info("Best validation accuracy: %.2f%%", best_acc * 100)
        if best_acc < 0.999:
            logger.warning(
                "Accuracy target (99.9%) not yet met. "
                "Record more samples and re-run: python train.py --collect"
            )
            return 1

    return 0


def _parse_device(value: str) -> int | str:
    """Convert device argument to int if numeric, else keep as string."""
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    sys.exit(main())
