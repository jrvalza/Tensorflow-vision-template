import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.utils.paths import get_evaluation_dir


def save_report(report: dict, filename: str) -> None:
    """Persist a metrics report as JSON in the current run's evaluation directory."""
    report_path = get_evaluation_dir() / filename
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


def save_figure(fig: Figure, filename: str) -> None:
    """Save a matplotlib figure into the current run's evaluation directory"""
    output_path = get_evaluation_dir() / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
