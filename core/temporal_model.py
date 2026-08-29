from __future__ import annotations

from pathlib import Path

import numpy as np

from core.config import SETTINGS


ACTION_CLASSES = [
    "none",
    "jab",
    "cross",
    "left_hook",
    "right_hook",
    "left_uppercut",
    "right_uppercut",
    "backfist",
    "spinning_backfist",
    "left_round_kick",
    "right_round_kick",
    "left_front_kick",
    "right_front_kick",
    "left_push_kick",
    "right_push_kick",
    "left_knee",
    "right_knee",
]


def build_temporal_network(architecture: str, input_dim: int, classes: int):
    """Create the checkpoint-compatible temporal network.

    `pose_transformer_v2` is the current training target. `gru_v1` remains
    supported so an older WarriorIQ checkpoint can still be loaded rather than
    silently becoming incompatible.
    """
    import torch
    import torch.nn as nn

    if architecture == "gru_v1":
        class GRUNet(nn.Module):
            def __init__(self):
                super().__init__()
                hidden = 192
                self.gru = nn.GRU(
                    input_dim,
                    hidden,
                    num_layers=2,
                    batch_first=True,
                    dropout=0.15,
                    bidirectional=True,
                )
                self.norm = nn.LayerNorm(hidden * 2)
                self.head = nn.Sequential(
                    nn.Linear(hidden * 2, 192),
                    nn.GELU(),
                    nn.Dropout(0.10),
                    nn.Linear(192, classes),
                )

            def forward(self, x):
                y, _ = self.gru(x)
                return self.head(self.norm(y[:, -1]))

        return GRUNet()

    if architecture != "pose_transformer_v2":
        raise ValueError(f"Unknown temporal architecture: {architecture}")

    class PoseTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            embed = 192
            self.input_norm = nn.LayerNorm(input_dim)
            self.input_projection = nn.Linear(input_dim, embed)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed))
            # Action windows are intentionally short for real-time inference;
            # keep extra room for future longer checkpoints.
            self.position = nn.Parameter(torch.zeros(1, 65, embed))
            layer = nn.TransformerEncoderLayer(
                d_model=embed,
                nhead=6,
                dim_feedforward=512,
                dropout=0.12,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=4, norm=nn.LayerNorm(embed))
            self.head = nn.Sequential(
                nn.LayerNorm(embed),
                nn.Linear(embed, 192),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(192, classes),
            )
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.position, std=0.02)

        def forward(self, x):
            b, t, _ = x.shape
            if t + 1 > self.position.shape[1]:
                raise ValueError(f"Sequence length {t} exceeds model capacity")
            y = self.input_projection(self.input_norm(x))
            cls = self.cls_token.expand(b, -1, -1)
            y = torch.cat([cls, y], dim=1)
            y = y + self.position[:, : t + 1]
            y = self.encoder(y)
            return self.head(y[:, 0])

    return PoseTransformer()


class TemporalModel:
    """Optional trained WarriorIQ kickboxing temporal classifier.

    A neural checkpoint is used only if a real trained checkpoint exists. Until
    WarriorIQ has a properly labeled kickboxing dataset with held-out complete
    fights, the analyzer falls back to its deterministic multi-frame action
    engine. Random/untrained weights are never exposed as analysis facts.
    """

    def __init__(self):
        self.available = False
        self.model = None
        self.device = None
        self.architecture = None
        self.validation = {}
        checkpoint = Path(SETTINGS.temporal_checkpoint)
        if not checkpoint.exists():
            return
        try:
            import torch

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            payload = torch.load(checkpoint, map_location=self.device)
            input_dim = int(payload.get("input_dim", 102)) if isinstance(payload, dict) else 102
            state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            architecture = payload.get("architecture", "gru_v1") if isinstance(payload, dict) else "gru_v1"
            classes = payload.get("classes", ACTION_CLASSES) if isinstance(payload, dict) else ACTION_CLASSES
            if list(classes) != ACTION_CLASSES:
                raise ValueError("Checkpoint action classes do not match this WarriorIQ build")
            self.model = build_temporal_network(architecture, input_dim, len(ACTION_CLASSES)).to(self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            self.architecture = architecture
            if isinstance(payload, dict):
                self.validation = {
                    "val_accuracy": payload.get("val_accuracy"),
                    "held_out_fights": payload.get("held_out_fights"),
                    "dataset_version": payload.get("dataset_version"),
                    "per_class_validation_accuracy": payload.get("per_class_validation_accuracy"),
                    "test_accuracy": payload.get("test_accuracy"),
                    "held_out_test_fights": payload.get("held_out_test_fights"),
                    "per_class_test_accuracy": payload.get("per_class_test_accuracy"),
                    "per_class_test_precision": payload.get("per_class_test_precision"),
                    "per_class_test_f1": payload.get("per_class_test_f1"),
                    "macro_test_f1": payload.get("macro_test_f1"),
                    "macro_action_test_f1": payload.get("macro_action_test_f1"),
                    "end_to_end_validation": payload.get("end_to_end_validation"),
                }
            self.available = True
        except Exception:
            self.available = False
            self.model = None

    def predict(self, sequence: np.ndarray) -> tuple[str, float] | None:
        if not self.available or self.model is None or sequence.ndim != 2:
            return None
        try:
            import torch

            tensor = torch.from_numpy(sequence.astype(np.float32))[None].to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(self.model(tensor), dim=-1)[0]
            confidence, index = torch.max(probs, dim=0)
            label = ACTION_CLASSES[int(index)]
            value = float(confidence)
            if label == "none" or value < SETTINGS.temporal_probability_threshold:
                return None
            return label, value
        except Exception:
            return None
