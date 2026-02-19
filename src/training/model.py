"""
training/model.py
-----------------
AlarmCNN – a compact binary convolutional classifier that operates on
log-mel-spectrograms produced by audio/features.py.

Architecture:
    Conv2d(1→32, 3×3) → BN → ReLU → MaxPool(2×2)
    Conv2d(32→64, 3×3) → BN → ReLU → MaxPool(2×2)
    Conv2d(64→128, 3×3) → BN → ReLU → AdaptiveAvgPool(4×4)
    Flatten → Linear(128×4×4 → 256) → ReLU → Dropout(0.3)
    Linear(256 → 1) → Sigmoid

Output: scalar in [0, 1].  Values ≥ 0.99 are treated as "alarm detected".
"""

from __future__ import annotations

import torch
import torch.nn as nn


DETECTION_THRESHOLD: float = 0.99  # confidence required to fire the alarm event


class AlarmCNN(nn.Module):
    """
    Small binary CNN classifier for alarm sound detection.

    Input
    -----
    Tensor of shape ``(batch, 1, N_MELS, T)`` – channel-first log-mel-spectrograms.

    Output
    ------
    Tensor of shape ``(batch, 1)`` – sigmoid probability.
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = self.features(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> float:
        """
        Return the scalar alarm probability for a single spectrogram tensor.

        Parameters
        ----------
        x:
            Tensor of shape ``(1, N_MELS, T)`` or ``(1, 1, N_MELS, T)``.

        Returns
        -------
        float
            Probability in [0, 1].
        """
        self.eval()
        with torch.no_grad():
            if x.ndim == 3:
                x = x.unsqueeze(0)   # add batch dimension
            return self.forward(x).item()

    def is_alarm(self, x: torch.Tensor, threshold: float = DETECTION_THRESHOLD) -> bool:
        """Return ``True`` when alarm confidence ≥ *threshold*."""
        return self.predict_proba(x) >= threshold
