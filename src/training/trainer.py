"""
training/trainer.py
-------------------
Training loop for AlarmCNN.

Trains until validation accuracy reaches TARGET_ACCURACY (99%) or
MAX_EPOCHS is exhausted.  The best model is saved to models/alarm_model.pt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.dataset import AlarmDataset, make_splits
from src.training.model import AlarmCNN

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "alarm_model.pt"

TARGET_ACCURACY: float = 0.99
MAX_EPOCHS: int = 200
BATCH_SIZE: int = 16
LEARNING_RATE: float = 1e-3
TRAIN_FRACTION: float = 0.8


def _accuracy(outputs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (outputs >= threshold).float()
    return (preds == labels.unsqueeze(1)).float().mean().item()


def train(
    n_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    target_accuracy: float = TARGET_ACCURACY,
    device_str: str = "cpu",
) -> Tuple[AlarmCNN, float]:
    """
    Train AlarmCNN on the collected sample data.

    Saves ``models/alarm_model.pt`` when validation accuracy reaches
    *target_accuracy*, then returns early.

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
    tuple[AlarmCNN, float]
        The trained model and the best validation accuracy achieved.
    """
    device = torch.device(device_str)

    # ── Dataset ───────────────────────────────────────────────────────────────
    full_dataset = AlarmDataset(augment=True)
    print(full_dataset.summary())

    train_ds, val_ds = make_splits(full_dataset, TRAIN_FRACTION)
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

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

    best_val_acc: float = 0.0
    best_epoch: int = 0

    print(f"\nTraining on {device}  (target val accuracy: {target_accuracy * 100:.0f}%)\n")
    print(f"{'Epoch':>6}  {'Train loss':>11}  {'Train acc':>10}  {'Val acc':>8}")
    print("-" * 44)

    for epoch in range(1, n_epochs + 1):
        # ── train ─────────────────────────────────────────────────────────────
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

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                outputs = model(specs)
                val_acc += _accuracy(outputs, labels) * len(specs)
        val_acc /= len(val_ds)

        scheduler.step(val_acc)

        print(f"{epoch:>6}  {train_loss:>11.4f}  {train_acc:>9.1%}  {val_acc:>7.1%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)

        if val_acc >= target_accuracy:
            print(
                f"\n✓ Target accuracy {target_accuracy * 100:.0f}% reached at epoch {epoch}."
                f"  Model saved to {MODEL_PATH}"
            )
            break
    else:
        print(
            f"\nTraining finished after {n_epochs} epochs. "
            f"Best val accuracy: {best_val_acc * 100:.2f}% (epoch {best_epoch})."
        )
        if best_val_acc < target_accuracy:
            print(
                "  Accuracy target not yet reached.\n"
                "  Tip: record more samples with:  python train.py --collect\n"
                "  Then re-run training:           python train.py --train"
            )
        print(f"  Best model saved to {MODEL_PATH}")

    return model, best_val_acc


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
