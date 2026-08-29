from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model_validation import audit_sequence_directory, classification_metrics
from core.temporal_model import ACTION_CLASSES, build_temporal_network


class SequenceDataset(Dataset):
    """Loads kickboxing pose sequences without leaking the same fight across splits.

    Each `.npz` must contain:
      x: float32 array shaped (T, 102)
      y: integer class index

    Strongly recommended:
      fight_id: string scalar identifying the source fight

    If fight_id is absent, the part of the filename before `__` is used. That
    keeps clips such as fight42__000123.npz grouped together.
    """

    def __init__(self, root: Path):
        self.items = sorted(root.glob("*.npz"))
        if not self.items:
            raise RuntimeError(f"No .npz sequences found in {root}")
        self.fight_ids = []
        self.labels = []
        for path in self.items:
            data = np.load(path, allow_pickle=False)
            if "fight_id" in data:
                fid = str(np.asarray(data["fight_id"]).item())
            else:
                fid = path.stem.split("__", 1)[0]
            self.fight_ids.append(fid)
            self.labels.append(int(data["y"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        data = np.load(self.items[index], allow_pickle=False)
        x = np.asarray(data["x"], dtype=np.float32)
        y = int(data["y"])
        if x.ndim != 2:
            raise ValueError(f"{self.items[index]} x must be 2D (T, features), got {x.shape}")
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def group_split(dataset: SequenceDataset, val_fraction: float, seed: int):
    fights = sorted(set(dataset.fight_ids))
    if len(fights) < 2:
        raise RuntimeError(
            "Need sequences from at least two distinct fights. A random clip split would leak the same fight into train and validation."
        )
    rng = random.Random(seed)
    rng.shuffle(fights)
    val_count = max(1, round(len(fights) * val_fraction))
    val_fights = set(fights[:val_count])
    train_indices = [i for i, f in enumerate(dataset.fight_ids) if f not in val_fights]
    val_indices = [i for i, f in enumerate(dataset.fight_ids) if f in val_fights]
    if not train_indices or not val_indices:
        raise RuntimeError("Fight-group split produced an empty train or validation set")
    return Subset(dataset, train_indices), Subset(dataset, val_indices), sorted(val_fights)


def evaluate(model, loader, device):
    model.eval()
    expected = []
    predicted = []
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(-1)
            expected.extend(ACTION_CLASSES[index] for index in y.detach().cpu().tolist())
            predicted.extend(ACTION_CLASSES[index] for index in pred.detach().cpu().tolist())
    return classification_metrics(expected, predicted, classes=ACTION_CLASSES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/sequences")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--architecture", choices=["pose_transformer_v2", "gru_v1"], default="pose_transformer_v2")
    parser.add_argument("--dataset-version", default="unversioned")
    parser.add_argument("--out", default="models/warrioriq_temporal_best.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_root = Path(args.data)
    audit = audit_sequence_directory(data_root)
    if audit["invalid_sequences"]:
        first = audit["issues"][0]
        raise RuntimeError(
            f"Dataset audit rejected {audit['invalid_sequences']} invalid sequence(s). "
            f"First problem: {first['file']}: {first['reason']}"
        )
    if not audit["experimental_train_ready"]:
        raise RuntimeError(
            "Experimental training requires at least 20 real actions, 20 negative-motion sequences "
            "and two separate fights. Run tools/audit_accuracy_dataset.py for the exact gaps."
        )
    dataset = SequenceDataset(data_root)
    sample_x, _ = dataset[0]
    input_dim = int(sample_x.shape[-1])
    train_ds, val_ds, val_fights = group_split(dataset, args.val_fraction, args.seed)
    training_fights = sorted(set(dataset.fight_ids) - set(val_fights))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_temporal_network(args.architecture, input_dim, len(ACTION_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Training sequences:", len(train_ds))
    print("Validation sequences:", len(val_ds))
    print("Held-out validation fights:", ", ".join(val_fights))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())

        validation_metrics = evaluate(model, val_loader, device)
        accuracy = float(validation_metrics["accuracy"] or 0.0)
        print(f"epoch {epoch:03d} loss={running/max(1,len(train_loader)):.4f} val_acc={accuracy:.4f}")
        if accuracy > best:
            best = accuracy
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "architecture": args.architecture,
                    "input_dim": input_dim,
                    "classes": ACTION_CLASSES,
                    "val_accuracy": best,
                    "held_out_fights": val_fights,
                    "training_fights": training_fights,
                    "dataset_version": args.dataset_version,
                    "per_class_validation_accuracy": {
                        name: item["recall"] for name, item in validation_metrics["per_class"].items()
                    },
                    "per_class_validation_precision": {
                        name: item["precision"] for name, item in validation_metrics["per_class"].items()
                    },
                    "per_class_validation_f1": {
                        name: item["f1"] for name, item in validation_metrics["per_class"].items()
                    },
                    "macro_validation_f1": validation_metrics["macro_f1"],
                    "macro_action_validation_f1": validation_metrics["macro_action_f1"],
                },
                out,
            )
            print("  saved", out)

    metrics_path = out.with_suffix(".validation.json")
    metrics_path.write_text(
        json.dumps(
            {
                "best_validation_accuracy": best,
                "held_out_validation_fights": val_fights,
                "dataset_version": args.dataset_version,
                "architecture": args.architecture,
                "warning": "Production acceptance still requires a separate untouched full-fight test set.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Best fight-group validation accuracy:", best)
    print("Next: evaluate on a separate untouched full-fight test set before promoting the checkpoint to production.")


if __name__ == "__main__":
    main()
