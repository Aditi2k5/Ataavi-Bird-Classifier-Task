from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def compute_and_save_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: List[str],
    out_dir: str,
    prefix: str = "",
    slice_masks: Optional[Dict[str, Sequence[bool]]] = None,
    title_suffix: str = "",
) -> dict:
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score, accuracy_score)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    idx = list(range(len(class_names)))

    report_txt = classification_report(y_true, y_pred, labels=idx,
                                       target_names=class_names, digits=3,
                                       zero_division=0)
    rep = classification_report(y_true, y_pred, labels=idx,
                                target_names=class_names, output_dict=True,
                                zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": {c: float(rep[c]["f1-score"]) for c in class_names},
        "per_class_support": {c: int(rep[c]["support"]) for c in class_names},
    }
    if slice_masks:
        metrics["slices"] = {}
        for name, mask in slice_masks.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            metrics["slices"][name] = {
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
                "macro_f1": float(f1_score(y_true[mask], y_pred[mask],
                                           average="macro", zero_division=0)),
            }

    cm = confusion_matrix(y_true, y_pred, labels=idx)
    plot_confusion(cm, class_names, out / f"{prefix}confusion_matrix.png",
                   title=f"Confusion matrix{(' — ' + title_suffix) if title_suffix else ''}")

    (out / f"{prefix}classification_report.txt").write_text(
        report_txt + "\n\nSummary:\n" +
        json.dumps({k: v for k, v in metrics.items() if k != "per_class_support"}, indent=2))
    (out / f"{prefix}metrics.json").write_text(json.dumps(metrics, indent=2))

    print(report_txt)
    print(f"\nmacro-F1 = {metrics['macro_f1']:.3f}   accuracy = {metrics['accuracy']:.3f}")
    if metrics.get("slices"):
        print("slices:", json.dumps(metrics["slices"], indent=2))
    print(f"[saved] {out / (prefix + 'confusion_matrix.png')}")
    return metrics


def plot_confusion(cm: np.ndarray, class_names: List[str], path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(class_names),
                                    1.4 + 1.0 * len(class_names)))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="viridis", cbar=True,
                xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1,
                ax=ax, annot_kws={"color": "white", "fontsize": 9})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title + "\n(cells = counts, colour = row-normalised recall)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
