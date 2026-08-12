import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from tensorflow.keras.callbacks import History

from src.utils.paths import get_evaluation_dir


def save_figure(fig: Figure, filename: str) -> None:
    """Save a matplotlib figure into the current run's evaluation directory"""
    output_path = get_evaluation_dir() / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: History) -> None:
    """Plot train vs validation curves for every metric in history, saved as training_curves.png."""

    metrics = [metric for metric in history.history if not metric.startswith("val_")]

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(6 * len(metrics), 5), sharex=True
    )

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):

        ax.plot(history.history[metric], label="Training", linewidth=1)

        val_metric = f"val_{metric}"
        if val_metric in history.history:
            ax.plot(history.history[val_metric], label="Validation", linewidth=1)

        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())

        if "accuracy" in metric:
            ax.set_ylim(0, 1.1)

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    fig.suptitle("Training History", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "training_curves.png")


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    """Plot a confusion matrix heatmap, saved as confusion_matrix.png."""
    fig, ax = plt.subplots(
        figsize=(max(6, len(class_names)), max(5, len(class_names) * 0.8)),
        constrained_layout=True,
    )

    im = ax.imshow(cm, cmap="Blues")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("True Label")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Samples")
    save_figure(fig, "confusion_matrix.png")
