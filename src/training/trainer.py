"""
training/trainer.py
-------------------
Training loop for AlarmCNN.

Trains until validation accuracy reaches TARGET_ACCURACY (99.9%) or
MAX_EPOCHS is exhausted.  The best model is saved to
``models/alarm_model_YYYY-MM-DD.pt`` with a companion metadata JSON file.
A symlink ``models/alarm_model.pt`` always points to the latest model.
"""

from __future__ import annotations

import datetime
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.audio.capture import SAMPLE_RATE
from src.audio.features import N_MELS, N_FFT, HOP_LENGTH, F_MIN, F_MAX
from src.training.collector import CLIP_DURATION
from src.training.dataset import AlarmDataset, make_splits
from src.training.model import AlarmCNN, DETECTION_THRESHOLD

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "alarm_model.pt"  # symlink target for load_model()

TARGET_ACCURACY: float = 0.999
MAX_EPOCHS: int = 200
BATCH_SIZE: int = 16
LEARNING_RATE: float = 1e-3
TRAIN_FRACTION: float = 0.8


def _accuracy(outputs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (outputs >= threshold).float()
    return (preds == labels.unsqueeze(1)).float().mean().item()


# ── helpers ───────────────────────────────────────────────────────────────────


def _dated_model_paths(models_dir: Path) -> Tuple[Path, Path]:
    """Return ``(model_path, meta_path)`` using today's date.

    If files for today already exist, appends ``_2``, ``_3``, etc.
    """
    date_str = datetime.date.today().isoformat()
    base = f"alarm_model_{date_str}"

    model_path = models_dir / f"{base}.pt"
    if not model_path.exists():
        return model_path, models_dir / f"{base}_meta.json"

    suffix = 2
    while True:
        candidate = models_dir / f"{base}_{suffix}.pt"
        if not candidate.exists():
            return candidate, models_dir / f"{base}_{suffix}_meta.json"
        suffix += 1


def _update_symlink(link: Path, target: Path) -> None:
    """Create or replace a symlink at *link* pointing to *target*."""
    relative = target.name  # same directory — use filename only
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(relative)


def _evaluate(
    model: AlarmCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run a full evaluation pass and return detailed metrics."""
    model.eval()
    total_loss = 0.0
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for specs, labels in loader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)
            total_loss += criterion(outputs, labels.unsqueeze(1)).item() * len(specs)
            preds = (outputs >= 0.5).float().squeeze(1)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "val_loss": round(total_loss / max(total, 1), 6),
        "val_accuracy": round((tp + tn) / max(total, 1), 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "false_positive_rate": round(fpr, 6),
        "false_negative_rate": round(fnr, 6),
    }


def _val_sample_counts(val_ds: Subset, full_dataset: AlarmDataset) -> Dict[str, int]:
    """Count positive/negative samples in a validation Subset."""
    n_pos = sum(1 for i in val_ds.indices if full_dataset._items[i][1] == 1)
    n_neg = len(val_ds) - n_pos
    return {"total": len(val_ds), "positive": n_pos, "negative": n_neg}


def _build_metadata(
    *,
    model_filename: str,
    created_date: str,
    training_duration_seconds: float,
    n_epochs: int,
    best_epoch: int,
    target_accuracy: float,
    target_reached: bool,
    full_dataset: AlarmDataset,
    train_ds: Subset,
    val_ds: Subset,
    performance: Dict[str, float],
    lr: float,
    batch_size: int,
    model: AlarmCNN,
) -> Dict[str, Any]:
    """Assemble the complete metadata dictionary."""
    file_paths = full_dataset.file_paths
    val_counts = _val_sample_counts(val_ds, full_dataset)

    return {
        "model_filename": model_filename,
        "created_date": created_date,
        "training_duration_seconds": round(training_duration_seconds, 2),
        "epochs": n_epochs,
        "best_epoch": best_epoch,
        "target_accuracy_reached": target_reached,
        "training_samples": {
            "total": len(full_dataset),
            "positive": full_dataset.n_positive,
            "negative": full_dataset.n_negative,
            "positive_files": file_paths["positive"],
            "negative_files": file_paths["negative"],
        },
        "validation_samples": val_counts,
        "model_performance": performance,
        "training_config": {
            "learning_rate": lr,
            "batch_size": batch_size,
            "optimizer": "Adam",
            "loss_function": "BCELoss",
            "scheduler": "ReduceLROnPlateau",
            "train_val_split": TRAIN_FRACTION,
            "target_accuracy": target_accuracy,
            "max_epochs": n_epochs,
        },
        "feature_config": {
            "sample_rate": SAMPLE_RATE,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "f_min": F_MIN,
            "f_max": F_MAX,
            "clip_duration_seconds": CLIP_DURATION,
        },
        "model_architecture": {
            "name": "AlarmCNN",
            "detection_threshold": DETECTION_THRESHOLD,
            "total_parameters": sum(p.numel() for p in model.parameters()),
        },
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
        },
        "notes": "",
    }


def train(
    n_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    target_accuracy: float = TARGET_ACCURACY,
    device_str: str = "cpu",
) -> Tuple[AlarmCNN, float, Dict[str, Any]]:
    """
    Train AlarmCNN on the collected sample data.

    Saves a dated model file (e.g. ``models/alarm_model_2026-03-17.pt``)
    alongside a metadata JSON file, and updates the ``models/alarm_model.pt``
    symlink to point to the new model.

    Parameters
    ----------
    n_epochs:
        Maximum number of epochs to train.
    batch_size:
        Mini-batch size.
    lr:
        Adam learning rate.
    target_accuracy:
        Stop and save when validation accuracy ≥ this value.
    device_str:
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    tuple[AlarmCNN, float, dict]
        The trained model, the best validation accuracy, and the metadata dict.
    """
    device = torch.device(device_str)

    # ── Dataset ───────────────────────────────────────────────────────────────
    full_dataset = AlarmDataset(augment=True)
    logger.info("%s", full_dataset.summary())

    train_ds, val_ds = make_splits(full_dataset, TRAIN_FRACTION)
    logger.info("Train: %d samples  |  Val: %d samples", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ── Model / optimiser ─────────────────────────────────────────────────────
    model = AlarmCNN().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=10
    )
    criterion = nn.BCELoss()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = _dated_model_paths(MODELS_DIR)

    best_val_acc: float = 0.0
    best_epoch: int = 0
    target_reached: bool = False

    logger.info("Training on %s  (target val accuracy: %.0f%%)", device, target_accuracy * 100)
    logger.info("%s", f"{'Epoch':>6}  {'Train loss':>11}  {'Train acc':>10}  {'Val acc':>8}")
    logger.info("%s", "-" * 44)

    t_start = time.time()
    interrupted = False

    try:
        for epoch in range(1, n_epochs + 1):
            # ── train ─────────────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            train_acc = 0.0
            for specs, labels in train_loader:
                specs, labels = specs.to(device), labels.to(device)
                optimiser.zero_grad()
                outputs = model(specs)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimiser.step()
                train_loss += loss.item() * len(specs)
                train_acc += _accuracy(outputs, labels) * len(specs)

            train_loss /= len(train_ds)
            train_acc /= len(train_ds)

            # ── validate ──────────────────────────────────────────────────────
            model.eval()
            val_acc = 0.0
            with torch.no_grad():
                for specs, labels in val_loader:
                    specs, labels = specs.to(device), labels.to(device)
                    outputs = model(specs)
                    val_acc += _accuracy(outputs, labels) * len(specs)
            val_acc /= len(val_ds)

            scheduler.step(val_acc)

            logger.info("%s", f"{epoch:>6}  {train_loss:>11.4f}  {train_acc:>9.1%}  {val_acc:>7.1%}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                _update_symlink(MODEL_PATH, model_path)

            if val_acc >= target_accuracy:
                target_reached = True
                logger.info(
                    "Target accuracy %.0f%% reached at epoch %d.  Model saved to %s",
                    target_accuracy * 100, epoch, model_path,
                )
                # break
        else:
            logger.info(
                "Training finished after %d epochs. Best val accuracy: %.2f%% (epoch %d).",
                n_epochs, best_val_acc * 100, best_epoch,
            )
            if best_val_acc < target_accuracy:
                logger.warning(
                    "Accuracy target not yet reached.\n"
                    "  Tip: record more samples with:  python train.py --collect\n"
                    "  Then re-run training:           python train.py --train"
                )
            logger.info("Best model saved to %s", model_path)

    except KeyboardInterrupt:
        interrupted = True
        logger.info(
            "Training interrupted at epoch %d. Best val accuracy: %.2f%% (epoch %d).",
            epoch, best_val_acc * 100, best_epoch,
        )

    training_duration = time.time() - t_start

    # ── Evaluate best checkpoint & save metadata ──────────────────────────────
    if best_epoch == 0:
        logger.warning("No checkpoint was saved (interrupted before first validation).")
        if model_path.exists():
            model_path.unlink()
        raise KeyboardInterrupt

    model.load_state_dict(
        torch.load(str(model_path), map_location=device, weights_only=True)
    )
    performance = _evaluate(model, val_loader, criterion, device)

    metadata = _build_metadata(
        model_filename=model_path.name,
        created_date=datetime.date.today().isoformat(),
        training_duration_seconds=training_duration,
        n_epochs=n_epochs,
        best_epoch=best_epoch,
        target_accuracy=target_accuracy,
        target_reached=target_reached,
        full_dataset=full_dataset,
        train_ds=train_ds,
        val_ds=val_ds,
        performance=performance,
        lr=lr,
        batch_size=batch_size,
        model=model,
    )

    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("Metadata saved to %s", meta_path)

    return model, best_val_acc, metadata


def load_model(model_path: Path = MODEL_PATH, device_str: str = "cpu") -> AlarmCNN:
    """
    Load a previously saved AlarmCNN from *model_path*.

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` file produced by :func:`train`.
    device_str:
        Device to map the weights to.

    Returns
    -------
    AlarmCNN
        Model in eval mode.
    """
    model = AlarmCNN()
    model.load_state_dict(
        torch.load(str(model_path), map_location=torch.device(device_str), weights_only=True)
    )
    model.eval()
    return model
