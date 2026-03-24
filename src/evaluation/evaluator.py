"""
Evaluator
=========
Runs inference on a DataLoader, computes classification metrics, and saves
diagnostic plots (confusion matrix, per-class F1 bar chart).

Metrics computed
----------------
  • Top-1 accuracy
  • Macro F1-score
  • Weighted F1-score
  • Per-class F1-score
  • Confusion matrix (absolute + normalised)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (works on Kaggle / headless)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate a classification model on a DataLoader.

    Parameters
    ----------
    model:
        Trained model (already on the correct device).
    data_loader:
        DataLoader to evaluate on (typically test split).
    class_names:
        Ordered list of class name strings.
    output_dir:
        Directory where plots and JSON results are saved.
    device:
        Target device; defaults to CUDA if available.
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        class_names: list[str],
        output_dir: str | Path = "outputs/evaluation",
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Run full evaluation.

        Returns
        -------
        dict
            ``{"accuracy": float, "f1_macro": float, "f1_weighted": float,
               "f1_per_class": dict, "report": str}``
        """
        all_preds, all_labels = self._collect_predictions()
        results = self._compute_metrics(all_preds, all_labels)

        self._save_results(results)
        self._plot_confusion_matrix(all_labels, all_preds)
        self._plot_f1_per_class(results["f1_per_class"])

        logger.info(
            "Evaluation complete  accuracy=%.4f  f1_macro=%.4f",
            results["accuracy"],
            results["f1_macro"],
        )
        return results

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _collect_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []

        for images, labels in tqdm(
            self.data_loader, desc="Evaluating", dynamic_ncols=True
        ):
            images = images.to(self.device, non_blocking=True)
            logits = self.model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

        return np.array(all_preds), np.array(all_labels)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(
        self, preds: np.ndarray, labels: np.ndarray
    ) -> dict:
        accuracy = float(accuracy_score(labels, preds))
        f1_macro = float(f1_score(labels, preds, average="macro", zero_division=0))
        f1_weighted = float(
            f1_score(labels, preds, average="weighted", zero_division=0)
        )
        f1_per_class_arr = f1_score(
            labels, preds, average=None, zero_division=0
        )
        f1_per_class = {
            cls: float(f1_per_class_arr[i])
            for i, cls in enumerate(self.class_names)
            if i < len(f1_per_class_arr)
        }
        report = classification_report(
            labels,
            preds,
            target_names=self.class_names,
            digits=4,
            zero_division=0,
        )

        logger.info("\n%s", report)

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "f1_per_class": f1_per_class,
            "report": report,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_results(self, results: dict) -> None:
        save_dict = {k: v for k, v in results.items() if k != "report"}
        out_path = self.output_dir / "metrics.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(save_dict, fh, indent=2)
        logger.info("Metrics saved → %s", out_path)

        report_path = self.output_dir / "classification_report.txt"
        report_path.write_text(results["report"], encoding="utf-8")

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _plot_confusion_matrix(
        self, labels: np.ndarray, preds: np.ndarray
    ) -> None:
        cm = confusion_matrix(labels, preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        n = len(self.class_names)
        fig, axes = plt.subplots(1, 2, figsize=(max(14, n * 1.2), max(6, n * 0.8)))

        for ax, data, title, fmt in zip(
            axes,
            [cm, cm_norm],
            ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
            ["d", ".2f"],
        ):
            im = ax.imshow(data, cmap="Blues")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(self.class_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(self.class_names, fontsize=8)
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("True", fontsize=10)
            ax.set_title(title, fontsize=11)

            thresh = data.max() / 2.0
            for i in range(n):
                for j in range(n):
                    ax.text(
                        j, i,
                        format(data[i, j], fmt),
                        ha="center",
                        va="center",
                        color="white" if data[i, j] > thresh else "black",
                        fontsize=7,
                    )

        plt.tight_layout()
        out_path = self.output_dir / "confusion_matrix.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix saved → %s", out_path)

    def _plot_f1_per_class(self, f1_per_class: dict[str, float]) -> None:
        classes = list(f1_per_class.keys())
        scores = [f1_per_class[c] for c in classes]

        sorted_pairs = sorted(zip(scores, classes), reverse=True)
        scores_sorted = [p[0] for p in sorted_pairs]
        classes_sorted = [p[1] for p in sorted_pairs]

        fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.9), 5))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(classes)))
        bars = ax.barh(range(len(classes_sorted)), scores_sorted, color=colors)
        ax.set_yticks(range(len(classes_sorted)))
        ax.set_yticklabels(classes_sorted, fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("F1 Score", fontsize=10)
        ax.set_title("Per-class F1 Score", fontsize=12)
        ax.axvline(x=sum(scores) / len(scores), color="navy", linestyle="--",
                   linewidth=1.2, label=f"Macro avg ({sum(scores)/len(scores):.3f})")
        ax.legend(fontsize=9)

        for bar, score in zip(bars, scores_sorted):
            ax.text(
                score + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                va="center",
                fontsize=8,
            )

        plt.tight_layout()
        out_path = self.output_dir / "f1_per_class.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Per-class F1 chart saved → %s", out_path)

    # ── Training-curve plot (called from train script) ────────────────────────

    @staticmethod
    def plot_training_curves(
        history: dict[str, list[float]],
        output_dir: str | Path = "outputs/evaluation",
    ) -> None:
        """Save loss and accuracy training curves from a history dict."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        epochs = range(1, len(history["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Loss
        ax1.plot(epochs, history["train_loss"], label="Train Loss", marker=".")
        ax1.plot(epochs, history["val_loss"], label="Val Loss", marker=".")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(epochs, history["train_acc"], label="Train Acc", marker=".")
        ax2.plot(epochs, history["val_acc"], label="Val Acc", marker=".")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = output_dir / "training_curves.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Training curves saved → %s", out_path)
