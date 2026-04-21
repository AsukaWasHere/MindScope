"""
severity_scorer.py — Severity scoring for the MindScope pipeline.

Responsibility (ONLY):
    Compute a composite severity score [0.0, 1.0] for each post based on
    three independent signals, then normalise and add the `severity_score`
    column to the DataFrame.

Severity Score Formula — Design Rationale:
    ┌──────────────────────────────────────────────────────────────────┐
    │  severity = w1 * S_sentiment                                     │
    │           + w2 * S_length                                        │
    │           + w3 * S_keywords                                      │
    │                                                                  │
    │  where:                                                          │
    │    w1 = 0.40  (SENTIMENT_WEIGHT)                                 │
    │    w2 = 0.20  (LENGTH_WEIGHT)                                    │
    │    w3 = 0.40  (KEYWORD_WEIGHT)                                   │
    └──────────────────────────────────────────────────────────────────┘

    Signal 1 — S_sentiment:
        Converts VADER compound score to a 0→1 negativity scale.
        compound = -1.0 → S_sentiment = 1.0  (maximally negative)
        compound = +1.0 → S_sentiment = 0.0  (maximally positive)
        Formula: S_sentiment = (1 - compound) / 2

    Signal 2 — S_length:
        Longer posts often correlate with more detailed distress descriptions.
        Word count is capped at config.SEVERITY_LENGTH_CAP (500 words) then
        divided by that cap to get a [0, 1] score.
        Formula: S_length = min(word_count, cap) / cap

    Signal 3 — S_keywords:
        Counts substring matches of crisis/distress keywords from
        config.SEVERITY_KEYWORDS against the raw (lowercase) text.
        Normalised by dividing by the total keyword count so the score
        stays in [0, 1] regardless of how many keywords fired.
        Formula: S_keywords = matched_count / len(SEVERITY_KEYWORDS)
                 (clamped to [0, 1])

    Final score is clamped to [0.0, 1.0] and rounded to 4 decimal places.

Weight Justification:
    - Sentiment and keywords each carry 40% weight: both directly measure
      the emotional content of the text, which is the primary severity signal.
    - Length carries 20% weight: it is an indirect proxy (long posts often
      indicate sustained distress) but is not independently sufficient.

Public API:
    compute_severity(df)         → DataFrame with `severity_score` column
    get_severity_stats(df)       → per-class severity summary dict

Input:  pd.DataFrame with `clean_text`, `sentiment_score`, `label`
Output: Enriched DataFrame + reports/metrics/severity_stats.json
                            + reports/figures/severity_distribution.png
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import config
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

plt.rcParams.update({
    "figure.facecolor": "#0F1117", "axes.facecolor":  "#1A1D27",
    "axes.edgecolor":   "#2E3146", "axes.labelcolor": "#C8CCDA",
    "xtick.color":      "#C8CCDA", "ytick.color":     "#C8CCDA",
    "text.color":       "#C8CCDA", "grid.color":      "#2E3146",
    "grid.linestyle":   "--",      "grid.linewidth":  0.6,
    "font.family":      "monospace",
})

# Severity colormap: low = cool blue, high = alarm red
_SEV_CMAP = "RdYlBu_r"


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def compute_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and attach a composite severity score to each post.

    Requires `sentiment_score` column (output of analyze_sentiment()).
    Also operates on `clean_text` and `text` (for word count).

    Args:
        df (pd.DataFrame): Must contain `clean_text`, `sentiment_score`,
                           and `label` columns.

    Returns:
        pd.DataFrame: Original DataFrame with `severity_score` column added.
                      Score is a float in [0.0, 1.0] — higher = more severe.

    Example:
        >>> df = compute_severity(enriched_df)
        >>> df[["label", "severity_score"]].groupby("label").mean()
    """
    _validate(df)
    logger.info("=== Severity Scoring ===")

    df = df.copy()

    # ── Signal 1: Sentiment-derived negativity ────────────────────────────
    S_sentiment = df["sentiment_score"].apply(_sentiment_to_severity)
    logger.info(f"  S_sentiment  — mean={S_sentiment.mean():.4f}")

    # ── Signal 2: Text length (word count, normalised) ────────────────────
    word_counts = df[config.CLEAN_TEXT_COLUMN].str.split().str.len().fillna(0)
    S_length    = (word_counts.clip(upper=config.SEVERITY_LENGTH_CAP)
                   / config.SEVERITY_LENGTH_CAP)
    logger.info(f"  S_length     — mean={S_length.mean():.4f}")

    # ── Signal 3: Keyword presence score ──────────────────────────────────
    S_keywords = df[config.CLEAN_TEXT_COLUMN].apply(_keyword_score)
    logger.info(f"  S_keywords   — mean={S_keywords.mean():.4f}")

    # ── Weighted combination ───────────────────────────────────────────────
    raw = (
        config.SEVERITY_SENTIMENT_WEIGHT * S_sentiment
        + config.SEVERITY_LENGTH_WEIGHT  * S_length
        + config.SEVERITY_KEYWORD_WEIGHT * S_keywords
    )

    # Clamp to [0, 1] and round
    df["severity_score"] = raw.clip(0.0, 1.0).round(4)

    overall_mean = df["severity_score"].mean()
    logger.info(f"  Overall mean severity: {overall_mean:.4f}")

    return df


def get_severity_stats(df: pd.DataFrame) -> dict:
    """
    Compute per-class severity statistics and save outputs to disk.

    Statistics produced per class:
        mean_severity    — primary KPI (average score across all posts)
        median_severity  — robust central tendency
        high_severity_pct — % of posts with severity > 0.6 (threshold for concern)
        max_severity     — worst-case post in this class

    Args:
        df (pd.DataFrame): Output of compute_severity() — must have
                           `severity_score` and `label` columns.

    Returns:
        dict: Nested stats dict, also saved to severity_stats.json.

    Example:
        >>> stats = get_severity_stats(scored_df)
        >>> stats["most_severe_class"]
        'ptsd'
    """
    ensure_dir(config.METRICS_DIR)
    ensure_dir(config.FIGURES_DIR)

    logger.info("Computing per-class severity statistics...")
    HIGH_THRESHOLD = 0.6

    class_stats = {}
    for lbl in config.CLASSES:
        subset = df[df[config.LABEL_COLUMN] == lbl]["severity_score"].dropna()
        if subset.empty:
            continue
        class_stats[lbl] = {
            "mean_severity":     round(float(subset.mean()), 4),
            "median_severity":   round(float(subset.median()), 4),
            "std_severity":      round(float(subset.std()), 4),
            "high_severity_pct": round(100 * (subset > HIGH_THRESHOLD).mean(), 2),
            "max_severity":      round(float(subset.max()), 4),
            "n_posts":           int(len(subset)),
        }
        s = class_stats[lbl]
        logger.info(
            f"  {lbl:<12}  mean={s['mean_severity']:.4f}  "
            f"high%={s['high_severity_pct']:.1f}%  max={s['max_severity']:.4f}"
        )

    most_severe   = max(class_stats, key=lambda k: class_stats[k]["mean_severity"])
    least_severe  = min(class_stats, key=lambda k: class_stats[k]["mean_severity"])
    logger.info(f"\n  Most severe class  : {most_severe}")
    logger.info(f"  Least severe class : {least_severe}")

    stats = {
        "by_class":          class_stats,
        "most_severe_class": most_severe,
        "least_severe_class": least_severe,
        "high_threshold_used": HIGH_THRESHOLD,
    }

    with open(config.SEVERITY_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Severity stats saved: {config.SEVERITY_STATS_PATH}")

    _plot_severity_distribution(df, class_stats)

    return stats


# ─────────────────────────────────────────────
# PRIVATE SIGNAL FUNCTIONS
# ─────────────────────────────────────────────

def _sentiment_to_severity(compound: float) -> float:
    """
    Map VADER compound score [-1, +1] to a negativity-based severity [0, 1].

    Derivation:
        compound = -1.0 → severity = (1 - (-1)) / 2 = 1.0  (most severe)
        compound =  0.0 → severity = (1 -   0) / 2 = 0.5
        compound = +1.0 → severity = (1 -  +1) / 2 = 0.0  (least severe)

    Args:
        compound (float): VADER compound score.

    Returns:
        float: Negativity severity contribution in [0.0, 1.0].
    """
    return max(0.0, min(1.0, (1.0 - compound) / 2.0))


def _keyword_score(text: str) -> float:
    """
    Count how many crisis/distress keywords appear in the text.

    Matching is done as substring search on the lowercased text so that
    stemmed forms are caught (e.g., "suicid" matches "suicidal", "suicide").
    Score is normalised by the total keyword count so it stays in [0, 1].

    Args:
        text (str): Clean text string (from `clean_text` column).

    Returns:
        float: Keyword severity contribution in [0.0, 1.0].
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    lowered  = text.lower()
    matched  = sum(1 for kw in config.SEVERITY_KEYWORDS if kw in lowered)
    return min(1.0, matched / max(1, len(config.SEVERITY_KEYWORDS)))


def _validate(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are absent."""
    required = {config.CLEAN_TEXT_COLUMN, "sentiment_score", config.LABEL_COLUMN}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"compute_severity() requires: {required}. Missing: {missing}\n"
            "Run analyze_sentiment() first."
        )


# ─────────────────────────────────────────────
# PRIVATE VISUALISATION
# ─────────────────────────────────────────────

def _plot_severity_distribution(df: pd.DataFrame, class_stats: dict) -> None:
    """
    Save a 3-panel severity figure.

    Panel 1 — Overlaid KDE curves of severity score per class.
    Panel 2 — Horizontal bar chart of mean severity per class (ranked).
    Panel 3 — Scatter: sentiment_score vs severity_score (colour = class).

    Output: reports/figures/severity_distribution.png
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Severity Score Analysis — Reddit Mental Health Posts",
                 fontsize=14, y=1.01)

    # ── Panel 1: KDE per class ────────────────────────────────────────────
    ax = axes[0]
    for lbl in config.CLASSES:
        subset = df[df[config.LABEL_COLUMN] == lbl]["severity_score"]
        sns.kdeplot(
            subset, ax=ax, label=lbl,
            color=config.CLASS_COLORS.get(lbl, "#AAAAAA"),
            linewidth=2, fill=True, alpha=0.12,
        )
    ax.axvline(0.6, color="#F0C040", linewidth=1.0, linestyle="--",
               label="High threshold (0.6)")
    ax.set_xlabel("Severity Score")
    ax.set_ylabel("Density")
    ax.set_title("Severity Distribution per Class")
    ax.legend(framealpha=0.2, fontsize=8)
    ax.set_xlim(0, 1)

    # ── Panel 2: Ranked mean severity bar chart ───────────────────────────
    ax2 = axes[1]
    sorted_classes = sorted(
        config.CLASSES,
        key=lambda l: class_stats.get(l, {}).get("mean_severity", 0),
        reverse=True,
    )
    means  = [class_stats.get(l, {}).get("mean_severity", 0) for l in sorted_classes]
    colors = [config.CLASS_COLORS.get(l, "#AAAAAA") for l in sorted_classes]
    bars   = ax2.barh(sorted_classes, means, color=colors, alpha=0.85, height=0.55)
    for bar, val in zip(bars, means):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", ha="left", fontsize=9)
    ax2.set_xlabel("Mean Severity Score")
    ax2.set_title("Mean Severity per Class (Ranked)")
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    ax2.grid(axis="x")

    # ── Panel 3: Scatter sentiment vs severity ────────────────────────────
    ax3 = axes[2]
    sample = df.sample(min(3000, len(df)), random_state=config.RANDOM_SEED)
    for lbl in config.CLASSES:
        sub = sample[sample[config.LABEL_COLUMN] == lbl]
        ax3.scatter(
            sub["sentiment_score"], sub["severity_score"],
            c=config.CLASS_COLORS.get(lbl, "#AAAAAA"),
            alpha=0.25, s=8, label=lbl,
        )
    ax3.set_xlabel("VADER Sentiment Score")
    ax3.set_ylabel("Severity Score")
    ax3.set_title("Sentiment vs Severity (sample)")
    ax3.legend(markerscale=2, framealpha=0.2, fontsize=8)
    ax3.axhline(0.6, color="#F0C040", linewidth=0.8, linestyle="--")
    ax3.axvline(0,   color="#FFFFFF",  linewidth=0.6, linestyle=":")

    fig.tight_layout()
    fig.savefig(config.SEVERITY_FIGURE_PATH, dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Severity figure saved: {config.SEVERITY_FIGURE_PATH}")
