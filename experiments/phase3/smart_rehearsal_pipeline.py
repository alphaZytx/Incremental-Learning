"""Phase 3 Smart Rehearsal pipeline.

This module orchestrates the combined rehearsal + drift-monitoring workflow
intended for the end-semester milestone.  Compared to the phase 1 baseline and
phase 2 prototype, this script exposes a modular trainer with:

* configuration dataclasses for reproducible experiment manifests,
* pluggable drift detectors (ADWIN if available, else a moving-average fallback),
* explicit accounting of light vs. heavy rehearsal cost, and
* JSONL event logging for downstream comparative analysis.

The implementation remains lightweight enough to run on CPU-only laptops while
mirroring the control flow planned for the full Smart Rehearsal system.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

SPLITMNIST: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
)

TASK_REGISTRY: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "splitmnist": SPLITMNIST,
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleMLP(nn.Module):
    """Two-hidden-layer MLP suited for 28x28 grayscale images."""

    def __init__(self, hidden_size: int = 256, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.net(x)


class ExemplarBuffer:
    """Class-balanced exemplar memory with simple reservoir control."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.storage: Dict[int, List[Tensor]] = {}

    def __len__(self) -> int:
        return sum(len(samples) for samples in self.storage.values())

    def add(self, images: Tensor, labels: Tensor) -> None:
        if self.capacity <= 0:
            return
        for img, label in zip(images, labels):
            cls = int(label.item())
            self.storage.setdefault(cls, []).append(img.cpu())
        self._rebalance()

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        if len(self) == 0:
            raise ValueError("Buffer is empty")
        flat = [
            (cls, tensor)
            for cls, tensors in self.storage.items()
            for tensor in tensors
        ]
        chosen = random.sample(flat, k=min(batch_size, len(flat)))
        imgs = torch.stack([tensor for _, tensor in chosen])
        labels = torch.tensor([cls for cls, _ in chosen], dtype=torch.long)
        return imgs, labels

    def items(self) -> Iterator[Tuple[Tensor, int]]:
        for cls, tensors in self.storage.items():
            for tensor in tensors:
                yield tensor, cls

    def _rebalance(self) -> None:
        if len(self) <= self.capacity:
            return
        per_class = max(1, self.capacity // max(len(self.storage), 1))
        for cls, tensors in list(self.storage.items()):
            random.shuffle(tensors)
            self.storage[cls] = tensors[:per_class]
            if not self.storage[cls]:
                del self.storage[cls]


class BufferDataset(Dataset[Tuple[Tensor, int]]):
    """Dataset wrapper to iterate over stored exemplars."""

    def __init__(self, buffer: ExemplarBuffer) -> None:
        self.samples = list(buffer.items())

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:  # type: ignore[override]
        tensor, label = self.samples[index]
        return tensor, label


class DriftDetector:
    """Wrapper around ADWIN with a moving-average fallback."""

    def __init__(self, delta: float, min_window: int = 20) -> None:
        self.delta = delta
        self.min_window = min_window
        try:
            from river.drift import ADWIN  # type: ignore

            self._backend = ADWIN(delta=delta)
            self._use_adwin = True
        except Exception:  # pragma: no cover - optional dependency
            self._backend = None
            self._use_adwin = False
        self._window: List[float] = []

    def update(self, value: float) -> bool:
        if self._use_adwin and self._backend is not None:
            in_drift, _ = self._backend.update(value)
            return bool(in_drift)

        # Fallback: detect a drop > delta in rolling mean
        self._window.append(value)
        if len(self._window) < self.min_window:
            return False
        recent = self._window[-self.min_window :]
        history = self._window[:-self.min_window]
        if not history:
            return False
        hist_mean = sum(history) / len(history)
        recent_mean = sum(recent) / len(recent)
        drop = hist_mean - recent_mean
        triggered = drop > self.delta
        if triggered:
            # reset window to avoid repeated triggers
            self._window = recent[-self.min_window // 2 :]
        return triggered


@dataclass
class ExperimentConfig:
    dataset: str = "splitmnist"
    buffer_size: int = 300
    batch_size: int = 64
    epochs: int = 2
    rehearsal_interval: int = 2
    full_rehearsal_epochs: int = 1
    detector_delta: float = 0.03
    detector_min_window: int = 20
    lr: float = 1e-3
    seed: int = 42
    device: str = "cpu"
    output_dir: Path = Path("outputs/phase3")
    eval_interval: int = 1
    hidden_size: int = 256

    def tasks(self) -> Tuple[Tuple[int, int], ...]:
        try:
            return TASK_REGISTRY[self.dataset.lower()]
        except KeyError as exc:  # pragma: no cover - guard for CLI misuse
            raise ValueError(f"Unknown dataset: {self.dataset}") from exc


@dataclass
class TrainingStats:
    task_id: int
    epoch: int
    phase: str
    accuracy: float
    loss: float
    buffer_size: int
    detector_triggered: bool = False
    notes: Dict[str, float] = field(default_factory=dict)


class EventLogger:
    """Writes experiment events to JSONL for later aggregation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf8")

    def log(self, stats: TrainingStats) -> None:
        payload = asdict(stats)
        json.dump(payload, self._fh)
        self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Core trainer
# ---------------------------------------------------------------------------


class SmartRehearsalTrainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = SimpleMLP(hidden_size=config.hidden_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.buffer = ExemplarBuffer(config.buffer_size)
        self.detector = DriftDetector(
            delta=config.detector_delta, min_window=config.detector_min_window
        )
        self.logger = EventLogger(config.output_dir / "events.jsonl")
        self.summary: Dict[str, float] = {
            "light_rehearsal_steps": 0.0,
            "heavy_rehearsal_epochs": 0.0,
        }

    # ------------------------------
    # Dataset handling
    # ------------------------------

    def _prepare_tasks(self) -> List[Tuple[Subset[Dataset], Subset[Dataset]]]:
        transform = transforms.ToTensor()
        train = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform,
        )
        test = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transform,
        )
        tasks: List[Tuple[Subset[Dataset], Subset[Dataset]]] = []
        for start, end in self.config.tasks():
            mask_train = [start <= label <= end for _, label in train]
            mask_test = [start <= label <= end for _, label in test]
            train_indices = [i for i, keep in enumerate(mask_train) if keep]
            test_indices = [i for i, keep in enumerate(mask_test) if keep]
            tasks.append(
                (
                    Subset(train, train_indices),
                    Subset(test, test_indices),
                )
            )
        return tasks

    # ------------------------------
    # Training / evaluation routines
    # ------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(labels)

            # Light rehearsal: interleave mini-buffer batches
            if len(self.buffer) > 0:
                try:
                    buf_images, buf_labels = self.buffer.sample(
                        self.config.batch_size // 2
                    )
                except ValueError:
                    buf_images = buf_labels = None
                if buf_images is not None:
                    self.summary["light_rehearsal_steps"] += 1
                    buf_images = buf_images.to(self.device)
                    buf_labels = buf_labels.to(self.device)
                    self.optimizer.zero_grad()
                    logits = self.model(buf_images)
                    loss = self.criterion(logits, buf_labels)
                    loss.backward()
                    self.optimizer.step()
        return total_loss / max(1, len(loader.dataset))

    def _evaluate(self, loader: Optional[DataLoader]) -> float:
        if loader is None:
            return 0.0
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        return correct / max(1, total)

    def _replay_buffer(self) -> Optional[DataLoader]:
        if len(self.buffer) == 0:
            return None
        dataset = BufferDataset(self.buffer)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

    def _ingest_task_samples(self, loader: DataLoader) -> None:
        # Take one batch per label to refresh exemplars
        for images, labels in loader:
            self.buffer.add(images.cpu(), labels.cpu())
            break  # only use first batch as exemplar refresh

    def _heavy_rehearsal(self) -> None:
        replay_loader = self._replay_buffer()
        if replay_loader is None:
            return
        self.summary["heavy_rehearsal_epochs"] += self.config.full_rehearsal_epochs
        for _ in range(self.config.full_rehearsal_epochs):
            self._train_epoch(replay_loader)

    # ------------------------------
    # Public API
    # ------------------------------

    def run(self) -> None:
        tasks = self._prepare_tasks()
        set_seed(self.config.seed)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        for task_id, (train_task, test_task) in enumerate(tasks):
            train_loader = DataLoader(
                train_task,
                batch_size=self.config.batch_size,
                shuffle=True,
            )
            test_loader = DataLoader(test_task, batch_size=self.config.batch_size)

            for epoch in range(1, self.config.epochs + 1):
                loss = self._train_epoch(train_loader)
                buffer_loader = self._replay_buffer()
                acc = self._evaluate(buffer_loader)
                triggered = False
                if (epoch % self.config.eval_interval == 0) and buffer_loader is not None:
                    triggered = self.detector.update(acc)
                    if triggered:
                        self._heavy_rehearsal()
                        acc = self._evaluate(buffer_loader)

                stats = TrainingStats(
                    task_id=task_id,
                    epoch=epoch,
                    phase="train",
                    accuracy=acc,
                    loss=loss,
                    buffer_size=len(self.buffer),
                    detector_triggered=triggered,
                )
                self.logger.log(stats)

            # Refresh exemplars after finishing the task
            self._ingest_task_samples(train_loader)

            # Evaluate on the current task's test split
            task_acc = self._evaluate(test_loader)
            self.logger.log(
                TrainingStats(
                    task_id=task_id,
                    epoch=self.config.epochs,
                    phase="eval",
                    accuracy=task_acc,
                    loss=0.0,
                    buffer_size=len(self.buffer),
                )
            )

        self._finalize()

    def _finalize(self) -> None:
        summary_path = self.config.output_dir / "summary.json"
        with summary_path.open("w", encoding="utf8") as fh:
            json.dump(self.summary, fh, indent=2)
        self.logger.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3 Smart Rehearsal pipeline")
    parser.add_argument("--dataset", type=str, default="splitmnist")
    parser.add_argument("--buffer-size", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--full-rehearsal-epochs", type=int, default=1)
    parser.add_argument("--detector-delta", type=float, default=0.03)
    parser.add_argument("--detector-min-window", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase3"))
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=256)
    return parser


def main(args: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args=args)
    config = ExperimentConfig(
        dataset=parsed.dataset,
        buffer_size=parsed.buffer_size,
        batch_size=parsed.batch_size,
        epochs=parsed.epochs,
        full_rehearsal_epochs=parsed.full_rehearsal_epochs,
        detector_delta=parsed.detector_delta,
        detector_min_window=parsed.detector_min_window,
        lr=parsed.lr,
        seed=parsed.seed,
        device=parsed.device,
        output_dir=parsed.output_dir,
        eval_interval=parsed.eval_interval,
        hidden_size=parsed.hidden_size,
    )
    trainer = SmartRehearsalTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
