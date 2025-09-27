"""Phase 2 adaptive rehearsal experiment for Smart Rehearsal project.

This module extends the SplitMNIST rehearsal baseline by wiring a drift
monitor (ADWIN or a lightweight fallback) that triggers an intensive replay
cycle whenever validation accuracy on the exemplar buffer degrades. The goal
is to demonstrate how adaptive control can save computation while sustaining
accuracy, providing a bridge toward the end-semester prototype.

Example usage::

    python -m experiments.phase2.adaptive_rehearsal --epochs 3 --buffer-size 300 \\
        --full-rehearsal-epochs 2 --detector-delta 0.01 --plot

Outputs are saved to the ``outputs/phase2`` directory by default.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

TASKS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
)


def set_seed(seed: int = 13) -> None:
    """Ensure deterministic behaviour for reproducibility."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    """Same light-weight classifier as the phase 1 baseline."""

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

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class RehearsalBuffer:
    """Fixed-capacity exemplar buffer storing class-balanced samples."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.storage: Dict[int, List[Tensor]] = {}

    def __len__(self) -> int:
        return sum(len(samples) for samples in self.storage.values())

    def add_samples(self, images: Tensor, labels: Tensor) -> None:
        if self.capacity <= 0:
            return
        for img, label in zip(images, labels):
            cls = int(label.item())
            self.storage.setdefault(cls, []).append(img.cpu())
        self._reduce()

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        if self.capacity <= 0 or len(self) == 0:
            raise ValueError("Buffer is empty")
        population = [
            (cls, tensor)
            for cls, samples in self.storage.items()
            for tensor in samples
        ]
        chosen = random.sample(population, k=min(batch_size, len(population)))
        images = [tensor.unsqueeze(0) for _, tensor in chosen]
        labels = [torch.tensor([cls], dtype=torch.long) for cls, _ in chosen]
        return torch.cat(images, dim=0), torch.cat(labels, dim=0)

    def iter_all(self) -> Iterable[Tuple[Tensor, Tensor]]:
        for cls, samples in self.storage.items():
            for tensor in samples:
                yield tensor, torch.tensor(cls, dtype=torch.long)

    def _reduce(self) -> None:
        total = len(self)
        if total <= self.capacity:
            return
        per_class = max(1, self.capacity // max(len(self.storage), 1))
        for cls, samples in list(self.storage.items()):
            random.shuffle(samples)
            self.storage[cls] = samples[:per_class]


class BufferDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Utility dataset to iterate over the exemplar memory."""

    def __init__(self, buffer: RehearsalBuffer) -> None:
        self.samples = list(buffer.iter_all())

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        img, label = self.samples[index]
        return img, label


class ADWINMonitor:
    """Wrapper that prefers River's ADWIN but falls back to heuristic drift detection."""

    def __init__(
        self,
        delta: float = 0.01,
        min_window: int = 5,
        fallback_drop: float = 0.05,
    ) -> None:
        self.min_window = min_window
        self.fallback_drop = fallback_drop
        self.history: List[float] = []
        try:
            from river.drift import ADWIN  # type: ignore
        except ImportError:
            self.detector = None
        else:
            self.detector = ADWIN(delta=delta)

    def update(self, value: float) -> bool:
        """Feed a new metric value; return True when drift is detected."""

        self.history.append(value)
        if self.detector is not None:
            return bool(self.detector.update(value))
        if len(self.history) < self.min_window + 1:
            return False
        prev_window = self.history[:-1][-self.min_window :]
        if not prev_window:
            return False
        previous_mean = sum(prev_window) / len(prev_window)
        drop = previous_mean - value
        return drop >= self.fallback_drop

    def reset(self) -> None:
        if self.detector is not None:
            self.detector.reset()
        self.history.clear()


@dataclass
class TrainingConfig:
    epochs: int = 3
    batch_size: int = 64
    buffer_size: int = 300
    lr: float = 1e-3
    full_rehearsal_epochs: int = 2
    detector_delta: float = 0.01
    detector_min_window: int = 5
    detector_fallback_drop: float = 0.05
    light_rehearsal_ratio: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: Path = Path("outputs/phase2")
    download: bool = True


@dataclass
class TaskMetrics:
    task_id: int
    seen_classes: Sequence[int]
    accuracy: float
    rehearsal_examples: int


@dataclass
class DriftEvent:
    task_id: int
    epoch: int
    buffer_accuracy: float
    rehearsal_steps: int


@dataclass
class TrainingSummary:
    metrics: List[TaskMetrics]
    events: List[DriftEvent]


def load_split_mnist(root: str, train: bool, download: bool) -> datasets.MNIST:
    transform = transforms.Compose([transforms.ToTensor()])
    return datasets.MNIST(root=root, train=train, download=download, transform=transform)


def filter_dataset(dataset: Dataset, classes: Tuple[int, int]) -> Subset:
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return Subset(dataset, indices)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class AdaptiveRehearsalTrainer:
    """Coordinates incremental learning with adaptive drift-triggered rehearsal."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = config.device
        self.model = MLP().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.buffer = RehearsalBuffer(config.buffer_size)
        self.monitor = ADWINMonitor(
            delta=config.detector_delta,
            min_window=config.detector_min_window,
            fallback_drop=config.detector_fallback_drop,
        )

    def train(self) -> TrainingSummary:
        set_seed()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = load_split_mnist("./data", train=True, download=self.config.download)
        test_dataset = load_split_mnist("./data", train=False, download=self.config.download)

        metrics: List[TaskMetrics] = []
        events: List[DriftEvent] = []

        for task_id, classes in enumerate(TASKS, start=1):
            subset = filter_dataset(train_dataset, classes)
            loader = make_loader(subset, self.config.batch_size, shuffle=True)

            for epoch in range(self.config.epochs):
                self.model.train()
                for images, labels in loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    if len(self.buffer) > 0 and random.random() < self.config.light_rehearsal_ratio:
                        try:
                            buf_images, buf_labels = self.buffer.sample(self.config.batch_size)
                        except ValueError:
                            pass
                        else:
                            images = torch.cat([images, buf_images.to(self.device)], dim=0)
                            labels = torch.cat([labels, buf_labels.to(self.device)], dim=0)

                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                buffer_acc = self._evaluate_buffer()
                if buffer_acc is not None:
                    triggered = self.monitor.update(buffer_acc)
                    if triggered:
                        rehearsal_steps = self._full_rehearsal()
                        events.append(
                            DriftEvent(
                                task_id=task_id,
                                epoch=epoch + 1,
                                buffer_accuracy=buffer_acc,
                                rehearsal_steps=rehearsal_steps,
                            )
                        )
                        self.monitor.reset()

            # add current task samples to the buffer once the task is finished
            all_images: List[Tensor] = []
            all_labels: List[Tensor] = []
            for batch_images, batch_labels in loader:
                all_images.append(batch_images)
                all_labels.append(batch_labels)
            if all_images:
                stacked_images = torch.cat(all_images, dim=0)
                stacked_labels = torch.cat(all_labels, dim=0)
                self.buffer.add_samples(stacked_images, stacked_labels)

            seen_classes = [cls for task_classes in TASKS[:task_id] for cls in task_classes]
            eval_subset = filter_dataset(test_dataset, tuple(seen_classes))
            eval_loader = make_loader(eval_subset, batch_size=256, shuffle=False)
            acc = evaluate(self.model, eval_loader, self.device)
            metrics.append(
                TaskMetrics(
                    task_id=task_id,
                    seen_classes=seen_classes,
                    accuracy=acc,
                    rehearsal_examples=len(self.buffer),
                )
            )

        return TrainingSummary(metrics=metrics, events=events)

    def _evaluate_buffer(self) -> float | None:
        if len(self.buffer) == 0:
            return None
        dataset = BufferDataset(self.buffer)
        loader = make_loader(dataset, batch_size=256, shuffle=False)
        return evaluate(self.model, loader, self.device)

    def _full_rehearsal(self) -> int:
        if len(self.buffer) == 0:
            return 0
        dataset = BufferDataset(self.buffer)
        loader = make_loader(dataset, batch_size=self.config.batch_size, shuffle=True)
        steps = 0
        for _ in range(self.config.full_rehearsal_epochs):
            self.model.train()
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                steps += 1
        return steps


def save_summary(summary: TrainingSummary, output_dir: Path) -> None:
    payload = {
        "metrics": [asdict(metric) for metric in summary.metrics],
        "events": [asdict(event) for event in summary.events],
    }
    json_file = output_dir / "phase2_summary.json"
    with json_file.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    csv_file = output_dir / "phase2_metrics.csv"
    with csv_file.open("w", encoding="utf-8") as fp:
        fp.write("task_id,seen_classes,accuracy,rehearsal_examples\n")
        for metric in summary.metrics:
            classes_str = "|".join(str(cls) for cls in metric.seen_classes)
            fp.write(
                f"{metric.task_id},{classes_str},{metric.accuracy:.4f},{metric.rehearsal_examples}\n"
            )

    events_file = output_dir / "phase2_events.csv"
    with events_file.open("w", encoding="utf-8") as fp:
        fp.write("task_id,epoch,buffer_accuracy,rehearsal_steps\n")
        for event in summary.events:
            fp.write(
                f"{event.task_id},{event.epoch},{event.buffer_accuracy:.4f},{event.rehearsal_steps}\n"
            )


def plot_summary(summary: TrainingSummary, config: TrainingConfig) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - plotting optional
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from exc

    output_path = config.output_dir / "phase2_accuracy.png"
    tasks = [metric.task_id for metric in summary.metrics]
    accuracies = [metric.accuracy for metric in summary.metrics]

    plt.figure(figsize=(7, 4))
    plt.plot(tasks, accuracies, marker="o", label="Accuracy")
    plt.ylim(0, 1.0)
    plt.xticks(tasks)
    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.title("Phase 2 Adaptive Rehearsal (SplitMNIST)")
    plt.grid(True, linestyle="--", alpha=0.4)

    if summary.events:
        event_x = [event.task_id - 1 + event.epoch / (config.epochs + 1) for event in summary.events]
        event_y = [event.buffer_accuracy for event in summary.events]
        plt.scatter(event_x, event_y, color="red", label="Drift Trigger")

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 adaptive rehearsal experiment")
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--buffer-size", type=int, default=TrainingConfig.buffer_size)
    parser.add_argument("--lr", type=float, default=TrainingConfig.lr)
    parser.add_argument(
        "--full-rehearsal-epochs",
        type=int,
        default=TrainingConfig.full_rehearsal_epochs,
        help="Epochs to run during a triggered rehearsal burst",
    )
    parser.add_argument("--detector-delta", type=float, default=TrainingConfig.detector_delta)
    parser.add_argument(
        "--detector-min-window",
        type=int,
        default=TrainingConfig.detector_min_window,
        help="Samples required before the fallback detector can fire",
    )
    parser.add_argument(
        "--detector-fallback-drop",
        type=float,
        default=TrainingConfig.detector_fallback_drop,
        help="Minimum accuracy drop to trigger in fallback mode",
    )
    parser.add_argument(
        "--light-rehearsal-ratio",
        type=float,
        default=TrainingConfig.light_rehearsal_ratio,
        help="Probability of mixing buffer samples in standard updates",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TrainingConfig.output_dir,
        help="Directory for metrics, events, and plots",
    )
    parser.add_argument("--no-download", action="store_true", help="Disable dataset download")
    parser.add_argument("--plot", action="store_true", help="Generate accuracy and drift plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        lr=args.lr,
        full_rehearsal_epochs=args.full_rehearsal_epochs,
        detector_delta=args.detector_delta,
        detector_min_window=args.detector_min_window,
        detector_fallback_drop=args.detector_fallback_drop,
        light_rehearsal_ratio=args.light_rehearsal_ratio,
        output_dir=args.output_dir,
        download=not args.no_download,
    )

    trainer = AdaptiveRehearsalTrainer(config)
    summary = trainer.train()
    save_summary(summary, config.output_dir)

    if args.plot:
        plot_path = plot_summary(summary, config)
        print(f"Saved phase 2 accuracy plot to: {plot_path}")

    print(
        "Stored metrics for %d tasks with %d drift events in %s"
        % (len(summary.metrics), len(summary.events), config.output_dir)
    )


if __name__ == "__main__":
    main()
