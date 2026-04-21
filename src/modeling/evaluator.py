"""
evaluator.py — Model evaluation for the MindScope pipeline.

Responsibility (ONLY):
    Accept a fitted model + test data, compute all metrics, generate
    confusion matrix figures, build the comparison table, and identify
    the best model. No training happens here.

Public API:
    evaluate_model(name, model, X_test, y_test)  → dict of metrics
    compare_models(results_dict, trained_models)  → comparison DataFrame
    save_evaluation_outputs(results, comparison)  → writes JSON/CSV/PNG

Input:  Fitted model, X_test (sparse), y_test (Series/array)
Output:
    reports/metrics/{name}_classification_report.txt
    reports/metrics/all_models_metrics.json
    reports/metrics/model_comparison.csv
    reports/figures/confusion_matrix_{name}.png
    reports/figures/model_comparison_chart.png
"""

import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils import config
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Consistent dark plot style (matches eda.py) ────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#1A1D27",
    "axes.edgecolor":   "#2E3146",
    "axes.labelcolor":  "#C8CCDA",
    "xtick.color":      "#C8CCDA",
    "ytick.color":      "#C8CCDA",
    "text.color":       "#C8CCDA",
    "grid.color":       "#2E3146",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "font.family":      "monospace",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})

# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """
    Compute all evaluation metrics for a single fitted model.

    Metrics computed:
        - Accuracy
        - Precision (macro-averaged)
        - Recall    (macro-averaged)
        - F1        (macro-averaged)
        - Per-class precision / recall / F1  (from classification_report)
        - Confusion matrix array

    Macro-averaging treats every class equally regardless of support size,
    which is appropriate here because class imbalance is expected.

    Args:
        name (str):   Human-readable model name (e.g., "logistic_regression").
        model:        Fitted sklearn estimator with a .predict() method.
        X_test:       Sparse feature matrix (from tfidf_vectorizer.py).
        y_test:       True string labels (pd.Series or array-like).

    Returns:
        dict: {
            "accuracy":  float,
            "precision": float,   # macro
            "recall":    float,   # macro
            "f1_macro":  float,
            "per_class": dict,    # from classification_report
            "confusion_matrix": list[list[int]]
        }

    Example:
        >>> result = evaluate_model("svm", svm_model, X_test, y_test)
        >>> result["f1_macro"]
        0.8234
    """
    logger.info(f"── Evaluating: {name}")

    y_pred = model.predict(X_test)

    accuracy  = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4)
    recall    = round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4)
    f1_macro  = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)

    report_dict = classification_report(
        y_test, y_pred,
        target_names=config.CLASSES,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_test, y_pred,
        target_names=config.CLASSES,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=config.CLASSES).tolist()

    result = {
        "accuracy":         accuracy,
        "precision":        precision,
        "recall":           recall,
        "f1_macro":         f1_macro,
        "per_class":        {k: v for k, v in report_dict.items()
                             if k in config.CLASSES},
        "confusion_matrix": cm,
    }

    # Log summary
    logger.info(f"   Accuracy  : {accuracy:.4f}")
    logger.info(f"   Precision : {precision:.4f}  (macro)")
    logger.info(f"   Recall    : {recall:.4f}  (macro)")
    logger.info(f"   F1 Macro  : {f1_macro:.4f}")

    # Save per-model text report
    _save_classification_report(name, report_str)

    # Save per-model confusion matrix figure
    _plot_confusion_matrix(name, np.array(cm))

    return result


def compare_models(
    results: dict,
    trained_models: dict,
) -> pd.DataFrame:
    """
    Build a comparison table from all model results, identify the winner,
    and save the best model artifacts.

    Comparison criterion: macro F1, because it treats every class equally
    and is robust to class imbalance — the most meaningful single metric
    for this 5-class problem.

    Args:
        results (dict):        {model_name: metrics_dict} from evaluate_model().
        trained_models (dict): {model_name: fitted_model} from train_all_models().

    Returns:
        pd.DataFrame: Rows = models, columns = accuracy/precision/recall/f1_macro.
                      Sorted descending by f1_macro.

    Example:
        >>> df = compare_models(results, trained_models)
        >>> df.head()
    """
    logger.info("=== Model Comparison ===")

    rows = []
    for name, metrics in results.items():
        rows.append({
            "model":     name,
            "accuracy":  metrics["accuracy"],
            "precision": metrics["precision"],
            "recall":    metrics["recall"],
            "f1_macro":  metrics["f1_macro"],
        })

    comparison = (
        pd.DataFrame(rows)
        .sort_values("f1_macro", ascending=False)
        .reset_index(drop=True)
    )

    # Log the table
    logger.info("\n" + comparison.to_string(index=False))

    # Identify best model
    best_name  = comparison.iloc[0]["model"]
    best_f1    = comparison.iloc[0]["f1_macro"]
    best_model = trained_models[best_name]

    logger.info(f"\n🏆 Best model: '{best_name}'  (Macro F1 = {best_f1:.4f})")
    _log_winner_explanation(best_name, comparison)

    # Save best model artifacts via trainer (avoids circular import by
    # calling the save helper directly)
    from src.modeling.trainer import save_best_model
    save_best_model(best_name, best_model)

    return comparison


def save_evaluation_outputs(results: dict, comparison: pd.DataFrame) -> None:
    """
    Persist all evaluation artifacts to reports/.

    Saves:
        reports/metrics/all_models_metrics.json
        reports/metrics/model_comparison.csv
        reports/figures/model_comparison_chart.png

    Args:
        results (dict):         Raw metrics dict from all evaluate_model() calls.
        comparison (DataFrame): Output of compare_models().

    Example:
        >>> save_evaluation_outputs(results, comparison_df)
    """
    ensure_dir(config.METRICS_DIR)
    ensure_dir(config.FIGURES_DIR)

    # JSON — full metrics for all models
    with open(config.ALL_METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Full metrics saved : {config.ALL_METRICS_PATH}")

    # CSV — comparison table
    comparison.to_csv(config.COMPARISON_TABLE_PATH, index=False)
    logger.info(f"Comparison table   : {config.COMPARISON_TABLE_PATH}")

    # Chart — visual comparison
    _plot_model_comparison(comparison)


# ─────────────────────────────────────────────
# PRIVATE HELPERS — LOGGING
# ─────────────────────────────────────────────

def _log_winner_explanation(best_name: str, comparison: pd.DataFrame) -> None:
    """
    Log a plain-English explanation of why the best model won.

    Compares the winner against every other model on all four metrics
    and writes an interpretable summary to the log.

    Args:
        best_name (str):        Name of the winning model.
        comparison (DataFrame): Full comparison table sorted by f1_macro.
    """
    explanations = {
        "logistic_regression": (
            "Logistic Regression works well on high-dimensional sparse TF-IDF "
            "features because it learns a linear decision boundary per class. "
            "With C=5 (mild regularisation) it avoids overfitting while "
            "capturing strong unigram/bigram signals in mental-health text."
        ),
        "naive_bayes": (
            "Multinomial Naive Bayes excels at short-to-medium text where "
            "individual word presence is highly discriminative. With alpha=0.1 "
            "(light smoothing) it assigns strong weights to class-specific "
            "vocabulary (e.g., 'flashback' → PTSD), making it fast and accurate."
        ),
        "svm": (
            "Linear SVM maximises the margin between classes in the TF-IDF "
            "feature space. It is particularly robust when classes overlap "
            "(e.g., depression vs. PTSD share vocabulary) because it focuses "
            "on the hardest examples near the boundary rather than the average."
        ),
    }

    best_row   = comparison[comparison["model"] == best_name].iloc[0]
    others     = comparison[comparison["model"] != best_name]

    logger.info("\n── Why did it win? ──")
    logger.info(explanations.get(best_name, "No explanation registered for this model."))

    for _, row in others.iterrows():
        delta_f1  = best_row["f1_macro"] - row["f1_macro"]
        delta_acc = best_row["accuracy"]  - row["accuracy"]
        logger.info(
            f"  vs {row['model']:<25} "
            f"ΔF1={delta_f1:+.4f}  ΔAcc={delta_acc:+.4f}"
        )


# ─────────────────────────────────────────────
# PRIVATE HELPERS — PERSISTENCE
# ─────────────────────────────────────────────

def _save_classification_report(name: str, report_str: str) -> None:
    """
    Write sklearn's text classification report to reports/metrics/.

    Args:
        name (str):       Model name used in filename.
        report_str (str): Formatted report string from classification_report().
    """
    path = config.METRICS_REPORT_PATH.format(name=name)
    with open(path, "w") as f:
        f.write(f"Classification Report — {name}\n")
        f.write("=" * 60 + "\n")
        f.write(report_str)
    logger.info(f"Report saved: {path}")


# ─────────────────────────────────────────────
# PRIVATE HELPERS — VISUALISATION
# ─────────────────────────────────────────────

def _plot_confusion_matrix(name: str, cm: np.ndarray) -> None:
    """
    Save a heatmap of the confusion matrix for one model.

    Cells are normalised by the true class (row) so the diagonal shows
    per-class recall — easier to read than raw counts on imbalanced data.

    Args:
        name (str):      Model name used in filename and title.
        cm (np.ndarray): Raw (un-normalised) confusion matrix, shape (5, 5).
    """
    # Normalise by row (true label) — shows recall per class
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(
            cm.sum(axis=1, keepdims=True) == 0,
            0,
            cm / cm.sum(axis=1, keepdims=True),
        )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Confusion Matrix — {name.replace('_', ' ').title()}",
        fontsize=14, y=1.01,
    )

    # Left: raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=config.CLASSES,
        yticklabels=config.CLASSES,
        ax=axes[0],
        cmap="Blues",
        linewidths=0.5,
        linecolor="#2E3146",
        annot_kws={"size": 10},
    )
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Right: row-normalised (recall view)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=config.CLASSES,
        yticklabels=config.CLASSES,
        ax=axes[1],
        cmap="Blues",
        vmin=0, vmax=1,
        linewidths=0.5,
        linecolor="#2E3146",
        annot_kws={"size": 10},
    )
    axes[1].set_title("Row-Normalised (Recall per Class)")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    fig.tight_layout()
    path = config.CONFUSION_MATRIX_PATH.format(name=name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Confusion matrix saved: {path}")


def _plot_model_comparison(comparison: pd.DataFrame) -> None:
    """
    Save a grouped bar chart comparing all models on four metrics.

    Models are on the x-axis; the best model's bars are highlighted
    with a brighter fill.

    Output: reports/figures/model_comparison_chart.png
    """
    metrics  = ["accuracy", "precision", "recall", "f1_macro"]
    labels   = ["Accuracy", "Precision", "Recall", "F1 Macro"]
    n_models = len(comparison)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    bar_w = 0.18

    # Colour per metric
    metric_colors = ["#5B8DB8", "#4FAF8C", "#E07B54", "#7A6BAE"]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Model Comparison — All Metrics", fontsize=14)

    for i, (metric, label, color) in enumerate(zip(metrics, labels, metric_colors)):
        vals   = comparison[metric].values
        offset = (i - n_metrics / 2 + 0.5) * bar_w
        bars   = ax.bar(x + offset, vals, width=bar_w, label=label,
                        color=color, alpha=0.85, zorder=2)

        # Annotate bar tops
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#C8CCDA",
            )

    # Highlight best model column
    best_x = comparison.index[comparison["model"] == comparison.iloc[0]["model"]][0]
    ax.axvspan(best_x - 0.45, best_x + 0.45, alpha=0.06, color="#FFFFFF", zorder=1)
    ax.text(
        best_x, 1.01,
        "🏆 Best",
        ha="center", va="bottom",
        fontsize=9, color="#F0C040",
        transform=ax.get_xaxis_transform(),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("_", "\n") for m in comparison["model"]],
        fontsize=9,
    )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", framealpha=0.2, fontsize=9)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    path = os.path.join(config.FIGURES_DIR, "model_comparison_chart.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Comparison chart saved: {path}")