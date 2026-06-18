"""
models.py — Neural network architectures for chatter detection.

Defined in a standalone module so that torch.load() can deserialize saved
model objects from any page (training, inference, model selector) without
import-cycle issues.

Architectures
─────────────
ChatterCNN      Pure 1-D CNN on OD waveform.  Backward-compatible with any
                .pt files trained before HybridChatterNet was introduced.

HybridChatterNet  Two-stream model:
                  • Fast Branch  — same 1-D CNN on OD / ovality waveform
                  • Context Branch — MLP on slow process variables
                  (air-ramp pressure, section footage, temperatures, etc.)
                After training, the normalizer statistics are stored directly
                on the model instance (ctx_mean, ctx_std) so that inference
                code can apply the same z-score transform without keeping a
                separate scaler file.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ── Context feature tags ──────────────────────────────────────────────────────
# Must match the WebSocket / XLSX tag names exactly.
CONTEXT_TAGS: list[str] = [
    "AirRampPressure_Val",              # ramps linearly with footage → key long-range signal
    "FtCounters_AirRampFootage_Total",  # absolute footage; normalize at inference time
    # "NDC_System_Ovality_Value",         # elevated during chatter
    "OilHeater_DeliveryTemp_F",
    "OilHeater_ReturnTemp_F",
    "PTs_PT_300_Val",
    "PTs_PT_400_Val",
]
N_CONTEXT = len(CONTEXT_TAGS)


# ── Architectures ─────────────────────────────────────────────────────────────

if _TORCH_OK:

    class ChatterCNN(nn.Module):
        """
        Baseline pure-CNN model (kept for backward compatibility).
        Input:  (batch, 1, window_size)
        Output: (batch, 2)  softmax → [no_chatter, chatter]
        """
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1,   32,  16), nn.BatchNorm1d(32),  nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(32,  64,   8), nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(64,  128,  4), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(128, 128,  4), nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.4), nn.Linear(128, 64), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(64,  2),  nn.Softmax(dim=1),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    # ─────────────────────────────────────────────────────────────────────────

    class HybridChatterNet(nn.Module):
        """
        Two-stream chatter detector that combines high-frequency OD shape with
        slow-varying process context (air-ramp pressure, section footage, …).

        Fast Branch   1-D CNN on z-score-normalised OD window
                      Input:  (batch, 1, window_size)  →  embedding (batch, 128)

        Context Branch  MLP on per-window mean of slow process variables
                        Input:  (batch, N_CONTEXT)  →  embedding (batch, 32)
                        Features are z-score normalised using stats stored on
                        the model (ctx_mean, ctx_std) after training.

        Classifier Head  concat → (batch, 160) → Linear → (batch, 2) softmax

        Normalizer stats are stored directly on the model instance so that a
        single torch.save / torch.load round-trip preserves everything needed
        for inference:
            model.ctx_mean   np.ndarray  shape (N_CONTEXT,)
            model.ctx_std    np.ndarray  shape (N_CONTEXT,)
        """

        def __init__(self):
            super().__init__()

            # ── Fast branch (identical depth to ChatterCNN) ──────────────────
            self.cnn = nn.Sequential(
                nn.Conv1d(1,   32,  16), nn.BatchNorm1d(32),  nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(32,  64,   8), nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(64,  128,  4), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(128, 128,  4), nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),                   # → (batch, 128)
            )

            # ── Context branch (MLP) ─────────────────────────────────────────
            self.mlp = nn.Sequential(
                nn.Linear(N_CONTEXT, 64), nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),        nn.ReLU(),
            )                                   # → (batch, 32)

            # ── Classifier head ──────────────────────────────────────────────
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(128 + 32, 64), nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2),
                nn.Softmax(dim=1),
            )

            # Placeholder normalizer stats (overwritten by training pipeline)
            self.ctx_mean: np.ndarray = np.zeros(N_CONTEXT, dtype=np.float32)
            self.ctx_std:  np.ndarray = np.ones(N_CONTEXT,  dtype=np.float32)

        def forward(self, x_od: "torch.Tensor", x_ctx: "torch.Tensor") -> "torch.Tensor":
            """
            x_od:  (batch, 1, window_size) — z-score normalised OD window
            x_ctx: (batch, N_CONTEXT)      — already normalised context vector
            Returns (batch, 2) softmax probabilities.
            """
            cnn_out = self.cnn(x_od)                        # (batch, 128)
            mlp_out = self.mlp(x_ctx)                       # (batch, 32)
            return self.head(torch.cat([cnn_out, mlp_out], dim=1))

        # Convenience: detect at runtime without importing this class elsewhere
        is_hybrid: bool = True

else:
    # Stubs so the rest of the codebase can import the names safely
    class ChatterCNN:          pass   # type: ignore
    class HybridChatterNet:    pass   # type: ignore
    HybridChatterNet.is_hybrid = True  # type: ignore
