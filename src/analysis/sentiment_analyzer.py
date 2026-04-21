"""
sentiment_analyzer.py — Sentiment analysis for the MindScope pipeline.

Responsibility (ONLY):
    Accept the processed DataFrame, apply VADER sentiment analysis to the
    `clean_text` column, produce `sentiment_score` and `sentiment_label`
    columns, compute per-class statistics, generate figures, and save outputs.

Why VADER over TextBlob?
    VADER (Valence Aware Dictionary and sEntiment Reasoner) was designed
    specifically for short social-media text. It understands:
      - Capitalisation intensity ("VERY SAD" scores more negative than "very sad")
      - Punctuation amplification ("hopeless!!!" > "hopeless")
      - Negation handling ("not happy" ≠ "happy")
      - Common social-media idioms
    These traits make it significantly more accurate than TextBlob on Reddit
    mental health posts where informal language and emotional intensity are common.

Public API:
    analyze_sentiment(df)  → enriched DataFrame with new columns
    get_sentiment_stats(df) → per-class summary statistics dict

New columns produced:
    sentiment_score  — VADER compound score, float in [-1.0, +1.0]
                       -1.0 = most negative, +1.0 = most positive
    sentiment_label  — categorical: "positive" / "neutral" / "negative"
                       thresholds defined in config.VADER_*_THRESH

Input:  pd.DataFrame with `clean_text` and `label` columns
Output: Enriched DataFrame, reports/metrics/sentiment_stats.json,
        reports/figures/sentiment_distribution.png
"""

import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from src.utils import config
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Consistent dark style ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0F1117", "axes.facecolor": "#1A1D27",
    "axes.edgecolor":   "#2E3146", "axes.labelcolor": "#C8CCDA",
    "xtick.color":      "#C8CCDA", "ytick.color":     "#C8CCDA",
    "text.color":       "#C8CCDA", "grid.color":      "#2E3146",
    "grid.linestyle":   "--",      "grid.linewidth":  0.6,
    "font.family":      "monospace",
})

# Sentiment label palette
_SENTIMENT_COLORS = {
    "positive": "#4FAF8C",
    "neutral":  "#5B8DB8",
    "negative": "#C0575A",
}

# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply VADER sentiment analysis to every row of `clean_text`.

    Adds two new columns to the DataFrame:
        sentiment_score  — VADER compound score [-1.0, +1.0].
                           Computed as a normalised weighted sum of
                           individual word valence scores in VADER's
                           4500-word lexicon.
        sentiment_label  — Categorical label derived from compound score:
                           >= +0.05  → "positive"
                           <= -0.05  → "negative"
                           else      → "neutral"
                           Thresholds from config.VADER_*_THRESH.

    Args:
        df (pd.DataFrame): Must contain `clean_text` and `label` columns.

    Returns:
        pd.DataFrame: Original DataFrame with two new columns appended.
                      Row order and index are preserved.

    Example:
        >>> df = pd.read_csv("data/processed/processed_data.csv")
        >>> enriched = analyze_sentiment(df)
        >>> enriched[["clean_text", "sentiment_score", "sentiment_label"]].head(3)
    """
    _validate(df)
    logger.info("=== Sentiment Analysis: VADER ===")

    sia = _load_vader()

    logger.info(f"Scoring {len(df):,} posts...")
    scores = df[config.CLEAN_TEXT_COLUMN].apply(
        lambda text: sia.polarity_scores(str(text))["compound"]
        if isinstance(text, str) else 0.0
    )

    df = df.copy()
    df["sentiment_score"] = scores.round(4)
    df["sentiment_label"] = df["sentiment_score"].apply(_label_from_score)

    # Log overall distribution
    dist = df["sentiment_label"].value_counts()
    logger.info("Overall sentiment distribution:")
    for lbl, n in dist.items():
        logger.info(f"  {lbl:<10} {n:>6,} ({100*n/len(df):.1f}%)")

    return df


def get_sentiment_stats(df: pd.DataFrame) -> dict:
    """
    Compute per-class and per-label sentiment statistics and save outputs.

    Statistics produced per class:
        mean_score    — average compound score (most interpretable metric)
        median_score  — median compound score (robust to outliers)
        pct_negative  — fraction of posts labelled "negative"
        pct_positive  — fraction of posts labelled "positive"
        pct_neutral   — fraction of posts labelled "neutral"

    Also identifies which class has the most negative mean sentiment.

    Args:
        df (pd.DataFrame): Output of analyze_sentiment() — must have
                           `sentiment_score`, `sentiment_label`, `label`.

    Returns:
        dict: Nested stats dict, also saved to sentiment_stats.json.

    Example:
        >>> stats = get_sentiment_stats(enriched_df)
        >>> stats["most_negative_class"]
        'ptsd'
    """
    ensure_dir(config.METRICS_DIR)
    ensure_dir(config.FIGURES_DIR)

    logger.info("Computing per-class sentiment statistics...")

    class_stats = {}
    for lbl in config.CLASSES:
        subset = df[df[config.LABEL_COLUMN] == lbl]
        if subset.empty:
            continue

        total      = len(subset)
        mean_sc    = round(float(subset["sentiment_score"].mean()), 4)
        median_sc  = round(float(subset["sentiment_score"].median()), 4)
        pct_neg    = round(100 * (subset["sentiment_label"] == "negative").sum() / total, 2)
        pct_pos    = round(100 * (subset["sentiment_label"] == "positive").sum() / total, 2)
        pct_neu    = round(100 * (subset["sentiment_label"] == "neutral").sum()  / total, 2)

        class_stats[lbl] = {
            "mean_score":   mean_sc,
            "median_score": median_sc,
            "pct_negative": pct_neg,
            "pct_positive": pct_pos,
            "pct_neutral":  pct_neu,
            "n_posts":      total,
        }
        logger.info(
            f"  {lbl:<12}  mean={mean_sc:+.4f}  "
            f"neg={pct_neg:.1f}%  pos={pct_pos:.1f}%  neu={pct_neu:.1f}%"
        )

    # Identify the most negative class (lowest mean compound score)
    most_negative = min(class_stats, key=lambda k: class_stats[k]["mean_score"])
    most_positive = max(class_stats, key=lambda k: class_stats[k]["mean_score"])

    logger.info(f"\n  Most NEGATIVE class: {most_negative} "
                f"(mean={class_stats[most_negative]['mean_score']:+.4f})")
    logger.info(f"  Most POSITIVE class: {most_positive} "
                f"(mean={class_stats[most_positive]['mean_score']:+.4f})")

    stats = {
        "by_class":           class_stats,
        "most_negative_class": most_negative,
        "most_positive_class": most_positive,
    }

    # Save JSON
    with open(config.SENTIMENT_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Sentiment stats saved: {config.SENTIMENT_STATS_PATH}")

    # Save figures
    _plot_sentiment_distribution(df)
    _plot_sentiment_heatmap(class_stats)

    return stats


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _load_vader() -> SentimentIntensityAnalyzer:
    """Download VADER lexicon if needed and return the analyser."""
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


def _label_from_score(score: float) -> str:
    """
    Convert a VADER compound score to a human-readable sentiment label.

    Thresholds recommended by the original VADER paper (Hutto & Gilbert 2014):
        compound >= +0.05  → "positive"
        compound <= -0.05  → "negative"
        else               → "neutral"

    Args:
        score (float): VADER compound score in [-1.0, +1.0].

    Returns:
        str: "positive", "negative", or "neutral".
    """
    if score >= config.VADER_POSITIVE_THRESH:
        return "positive"
    if score <= config.VADER_NEGATIVE_THRESH:
        return "negative"
    return "neutral"


def _validate(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are absent."""
    required = {config.CLEAN_TEXT_COLUMN, config.LABEL_COLUMN}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"analyze_sentiment() requires columns: {required}. Missing: {missing}")


def _plot_sentiment_distribution(df: pd.DataFrame) -> None:
    """
    Save a 3-panel figure showing sentiment distributions per class.

    Panel 1 — KDE of compound scores per class (overlaid curves).
    Panel 2 — Stacked bar chart of label proportions per class.
    Panel 3 — Box plot of compound score per class.

    Output: reports/figures/sentiment_distribution.png
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Sentiment Analysis — Reddit Mental Health Posts",
                 fontsize=14, y=1.01)

    # ── Panel 1: KDE per class ───────────────────────────────────────────
    ax = axes[0]
    for lbl in config.CLASSES:
        subset = df[df[config.LABEL_COLUMN] == lbl]["sentiment_score"]
        sns.kdeplot(
            subset, ax=ax, label=lbl,
            color=config.CLASS_COLORS.get(lbl, "#AAAAAA"),
            linewidth=2, fill=True, alpha=0.12,
        )
    ax.axvline(0, color="#FFFFFF", linewidth=0.8, linestyle=":")
    ax.set_xlabel("VADER Compound Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution per Class")
    ax.legend(framealpha=0.2, fontsize=8)

    # ── Panel 2: Stacked bar — label proportions per class ───────────────
    ax2 = axes[1]
    label_order = ["negative", "neutral", "positive"]
    bottom = np.zeros(len(config.CLASSES))
    for sent_lbl in label_order:
        vals = [
            (df[df[config.LABEL_COLUMN] == cls]["sentiment_label"] == sent_lbl).mean()
            for cls in config.CLASSES
        ]
        ax2.bar(
            config.CLASSES, vals, bottom=bottom,
            color=_SENTIMENT_COLORS[sent_lbl],
            label=sent_lbl, alpha=0.85,
        )
        bottom += np.array(vals)

    ax2.set_xlabel("Class")
    ax2.set_ylabel("Proportion")
    ax2.set_title("Sentiment Label Proportions")
    ax2.legend(framealpha=0.2, fontsize=8, loc="upper right")
    ax2.set_ylim(0, 1.05)

    # ── Panel 3: Box plots of compound score per class ────────────────────
    ax3 = axes[2]
    plot_data = [
        df[df[config.LABEL_COLUMN] == lbl]["sentiment_score"].dropna().values
        for lbl in config.CLASSES
    ]
    bp = ax3.boxplot(
        plot_data, labels=config.CLASSES, patch_artist=True,
        medianprops=dict(color="#FFFFFF", linewidth=2),
        whiskerprops=dict(color="#C8CCDA"),
        capprops=dict(color="#C8CCDA"),
        flierprops=dict(marker="o", markerfacecolor="#C8CCDA",
                        markersize=2, alpha=0.3, linestyle="none"),
    )
    for patch, lbl in zip(bp["boxes"], config.CLASSES):
        patch.set_facecolor(config.CLASS_COLORS.get(lbl, "#AAAAAA"))
        patch.set_alpha(0.75)
    ax3.axhline(0, color="#FFFFFF", linewidth=0.8, linestyle=":")
    ax3.set_xlabel("Class")
    ax3.set_ylabel("VADER Compound Score")
    ax3.set_title("Score Distribution (Box Plot)")
    ax3.grid(axis="y")

    fig.tight_layout()
    fig.savefig(config.SENTIMENT_FIGURE_PATH, dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Sentiment figure saved: {config.SENTIMENT_FIGURE_PATH}")


def _plot_sentiment_heatmap(class_stats: dict) -> None:
    """
    Save a heatmap of per-class mean sentiment score + pct_negative.

    Makes the most-negative class immediately visually obvious.
    Output: reports/figures/sentiment_heatmap.png
    """
    rows  = []
    index = []
    for lbl in config.CLASSES:
        if lbl not in class_stats:
            continue
        s = class_stats[lbl]
        rows.append([
            s["mean_score"],
            s["pct_negative"] / 100,
            s["pct_positive"] / 100,
        ])
        index.append(lbl)

    heat_df = pd.DataFrame(
        rows, index=index,
        columns=["Mean Score", "% Negative", "% Positive"],
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        heat_df, annot=True, fmt=".3f", ax=ax,
        cmap="RdYlGn", center=0,
        linewidths=0.5, linecolor="#2E3146",
        annot_kws={"size": 10},
    )
    ax.set_title("Per-Class Sentiment Summary", fontsize=13)
    fig.tight_layout()
    path = config.FIGURES_DIR + "/sentiment_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Sentiment heatmap saved: {path}")
