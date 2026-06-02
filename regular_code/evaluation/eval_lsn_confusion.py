import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from regular_code.config import LSN_CFG, PROJECT_ROOT
from regular_code.models.lsn import LSN
from regular_code.utils.device import get_device


class LSNEvalDataset(Dataset):
    def __init__(self, X: np.ndarray, K: np.ndarray, indices: np.ndarray):
        self.X = X[indices]
        self.K = K[indices]

    def __len__(self):
        return len(self.K)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).permute(2, 0, 1)
        y = torch.tensor(self.K[idx] - 1, dtype=torch.long)
        return x, y


def balanced_indices(K: np.ndarray, samples_per_class: Optional[int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []

    for k in sorted(np.unique(K)):
        idx = np.where(K == k)[0]
        if samples_per_class is not None and len(idx) > samples_per_class:
            idx = rng.choice(idx, size=samples_per_class, replace=False)
        selected.append(idx)

    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return indices


def build_confusion_matrix(model, loader, num_classes: int, device: torch.device) -> np.ndarray:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()

    with torch.no_grad():
        for x, y_true in loader:
            x = x.to(device)
            y_true = y_true.to(device)
            y_pred = torch.argmax(model(x), dim=1)

            for t, p in zip(y_true.cpu().numpy(), y_pred.cpu().numpy()):
                confusion[t, p] += 1

    return confusion


def plot_confusion_matrix(confusion: np.ndarray, save_path: Path):
    class_labels = [f"K={i}" for i in range(1, confusion.shape[0] + 1)]
    row_sum = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sum, out=np.zeros_like(confusion, dtype=np.float64), where=row_sum != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    warm_cmap = LinearSegmentedColormap.from_list(
        "warm_soft",
        ["#fffaf3", "#fdebd3", "#f8d2a8", "#eda36f", "#c96f4a"],
    )
    im = ax.imshow(normalized, cmap=warm_cmap, vmin=0, vmax=1)
    ax.set_facecolor("#fffaf3")
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted number of sources")
    ax.set_ylabel("True number of sources")
    ax.set_title("LSN Confusion Matrix")

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            text_color = "white" if normalized[i, j] > 0.62 else "#6b3a24"
            ax.text(
                j,
                i,
                f"{confusion[i, j]}\n{normalized[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Class-wise ratio")
    for spine in ax.spines.values():
        spine.set_color("#b77955")
        spine.set_linewidth(1.1)
    ax.tick_params(colors="#6b3a24")
    ax.xaxis.label.set_color("#6b3a24")
    ax.yaxis.label.set_color("#6b3a24")
    ax.title.set_color("#6b3a24")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Show LSN source-number prediction confusion matrix.")
    parser.add_argument("--samples-per-class", type=int, default=5000, help="Use 0 to evaluate all samples.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=Path, default=PROJECT_ROOT / "regular_code" / "evaluation" / "LSN_confusion_matrix.png")
    args = parser.parse_args()

    samples_per_class = None if args.samples_per_class == 0 else args.samples_per_class
    data = np.load(LSN_CFG.dataset_path)
    X = data["X"]
    K = data["K"]
    num_classes = X.shape[1] - 1

    indices = balanced_indices(K, samples_per_class, args.seed)
    dataset = LSNEvalDataset(X, K, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = get_device()
    model = LSN(num_sources=num_classes).to(device)
    model.load_state_dict(torch.load(LSN_CFG.checkpoint_path, map_location=device))

    confusion = build_confusion_matrix(model, loader, num_classes, device)
    total = confusion.sum()
    correct = np.trace(confusion)
    overall_acc = correct / total if total > 0 else 0.0
    per_class_acc = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)

    print(">>> LSN confusion matrix")
    print("Dataset:", LSN_CFG.dataset_path)
    print("Model:", LSN_CFG.checkpoint_path)
    print("Samples:", int(total))
    print("Overall accuracy: {:.4f}".format(overall_acc))
    for i, acc in enumerate(per_class_acc, start=1):
        print(f"K={i}: accuracy={acc:.4f}, samples={confusion[i - 1].sum()}")
    print("\nConfusion matrix rows=true K, columns=predicted K:")
    print(confusion)

    plot_confusion_matrix(confusion, args.save_path)
    print("Saved figure:", args.save_path)


if __name__ == "__main__":
    main()
