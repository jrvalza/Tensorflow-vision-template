import numpy as np
import matplotlib.pyplot as plt
from src.utils.savers import save_figure


class Plotter:
    """Provides plotting utilities for model evaluation."""

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: list[str]) -> None:
        """Plot and save a confusion matrix heatmap as confusion_matrix.png.

        Args:
            cm: Confusion matrix of shape (num_classes, num_classes).
            class_names: Names of the target classes.
        """
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
