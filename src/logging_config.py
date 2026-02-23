"""
src/logging_config.py
---------------------
Centralised logging configuration for ai-alarm-sound-recognition.

Call ``setup_logging()`` once at the start of each entry point (listen.py,
train.py, scripts/*).  All ``logging.getLogger(__name__)`` loggers in the
``src/`` package then automatically inherit both handlers.

Log format
----------
Both the console and file handler use:

    2026-02-23 14:05:01  INFO      src.inference.listener  Listening started

The file handler rotates at midnight, keeping 7 days of backups.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_file: str | Path,
    level: int = logging.INFO,
) -> None:
    """
    Configure the root logger to write to stdout *and* a rotating log file.

    Parameters
    ----------
    log_file:
        Path to the log file (e.g. ``"logs/listen.log"``).  The parent
        directory is created automatically if it does not exist.
    level:
        Root logger level (e.g. ``logging.DEBUG`` or ``logging.INFO``).
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT)

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # ── Rotating file handler ─────────────────────────────────────────────────
    file_handler = TimedRotatingFileHandler(
        filename=log_path,
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # ── Root logger ───────────────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if setup_logging is called more than once
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)
