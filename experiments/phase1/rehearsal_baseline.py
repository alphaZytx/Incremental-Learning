"""Phase 1 baseline experiments for the Smart Rehearsal project.

This module trains a simple multi-layer perceptron on SplitMNIST with
experience replay. The goal is to provide a reproducible baseline for the
mid-semester review where we can showcase rehearsal vs. non-rehearsal
performance and export accuracy curves for the report.

Example usage:
    python -m experiments.phase1.rehearsal_baseline --epochs 5 --buffer-size 200

Outputs are saved to the ``outputs/phase1`` directory by default.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

TASKS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    """Light-weight classifier for SplitMNIST."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RehearsalBuffer:
    """Fixed-capacity exemplar buffer storing (image, label) pairs."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.storage: Dict[int, List[torch.Tensor]] = {}

    def __len__(self) -> int:
        return sum(len(samples) for samples in self.storage.values())

    def add_samples(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        if self.capacity <= 0:
            return
        for img, label in zip(images, labels):
            label_int = int(label.item())
            self.storage.setdefault(label_int, []).append(img.cpu())
        self._reduce()

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.capacity <= 0 or len(self) == 0:
            raise ValueError("Buffer is empty")
        images = []
        labels = []
        available = [(cls, sample) for cls, samples in self.storage.items() for sample in samples]
        chosen = random.sample(available, k=min(batch_size, len(available)))
        for cls, sample in chosen:
            images.append(sample.unsqueeze(0))
            labels.append(torch.tensor([cls], dtype=torch.long))
        return torch.cat(images, dim=0), torch.cat(labels, dim=0)

    def iter_all(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        for cls, samples in self.storage.items():
            for sample in samples:
                yield sample, torch.tensor(cls, dtype=torch.long)

    def _reduce(self) -> None:
        total = len(self)
        if total <= self.capacity:
            return
        # simple class-balanced pruning
        per_class = max(1, self.capacity // max(len(self.storage), 1))
        for cls, samples in list(self.storage.items()):
            random.shuffle(samples)
            self.storage[cls] = samples[:per_class]


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 64
    buffer_size: int = 200
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: Path = Path("outputs/phase1")
    download: bool = True


@dataclass
class TaskMetrics:
    task_id: int
    seen_classes: List[int]
    accuracy: float
    rehearsal_examples: int


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


def rehearsal_training(config: TrainingConfig) -> List[TaskMetrics]:
    set_seed()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    device = config.device
    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    buffer = RehearsalBuffer(config.buffer_size)
    train_dataset = load_split_mnist("./data", train=True, download=config.download)
    test_dataset = load_split_mnist("./data", train=False, download=config.download)

    metrics: List[TaskMetrics] = []

    for task_id, classes in enumerate(TASKS, start=1):
        subset = filter_dataset(train_dataset, classes)
        loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True)

        for epoch in range(config.epochs):
            model.train()
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                if len(buffer) > 0:
                    try:
                        buf_images, buf_labels = buffer.sample(config.batch_size)
                    except ValueError:
                        buf_images = buf_labels = None
                    else:
                        buf_images = buf_images.to(device)
                        buf_labels = buf_labels.to(device)
                        images = torch.cat([images, buf_images], dim=0)
                        labels = torch.cat([labels, buf_labels], dim=0)

                logits = model(images)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # add samples from current task to buffer at end of each epoch
            all_images = []
            all_labels = []
            for batch_images, batch_labels in loader:
                all_images.append(batch_images)
                all_labels.append(batch_labels)
            buffer.add_samples(torch.cat(all_images, dim=0), torch.cat(all_labels, dim=0))

        seen_classes = [c for task_classes in TASKS[:task_id] for c in task_classes]
        eval_subset = filter_dataset(test_dataset, tuple(seen_classes))
        eval_loader = DataLoader(eval_subset, batch_size=256)
        acc = evaluate(model, eval_loader, device)
        metrics.append(
            TaskMetrics(
                task_id=task_id,
                seen_classes=seen_classes,
                accuracy=acc,
                rehearsal_examples=len(buffer),
            )
        )

    return metrics


def save_metrics(metrics: List[TaskMetrics], output_dir: Path) -> None:
    payload = [asdict(m) for m in metrics]
    metrics_file = output_dir / "phase1_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    # also create a CSV for quick inspection
    csv_file = output_dir / "phase1_metrics.csv"
    with csv_file.open("w", encoding="utf-8") as fp:
        fp.write("task_id,seen_classes,accuracy,rehearsal_examples\n")
        for m in metrics:
            classes_str = "|".join(str(c) for c in m.seen_classes)
            fp.write(f"{m.task_id},{classes_str},{m.accuracy:.4f},{m.rehearsal_examples}\n")


def plot_metrics(metrics: List[TaskMetrics], output_dir: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from exc
    tasks = [m.task_id for m in metrics]
    accuracies = [m.accuracy for m in metrics]

    plt.figure(figsize=(6, 4))
    plt.plot(tasks, accuracies, marker="o", label="Accuracy")
    plt.ylim(0, 1.0)
    plt.xticks(tasks)
    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.title("Phase 1 Rehearsal Baseline (SplitMNIST)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    output_path = output_dir / "phase1_accuracy.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 rehearsal baseline experiment")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per task")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--buffer-size", type=int, default=200, help="Total rehearsal buffer capacity")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase1"),
        help="Directory to store metrics and plots",
    )
    parser.add_argument("--no-download", action="store_true", help="Do not download datasets")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate accuracy plot for inclusion in the report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        lr=args.lr,
        output_dir=args.output_dir,
        download=not args.no_download,
    )

    metrics = rehearsal_training(config)
    save_metrics(metrics, config.output_dir)

    if args.plot:
        plot_path = plot_metrics(metrics, config.output_dir)
        print(f"Saved accuracy plot to: {plot_path}")

    print(f"Stored metrics for {len(metrics)} tasks in {config.output_dir}")


if __name__ == "__main__":
    main()
