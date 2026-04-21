"""
eda.py — Exploratory Data Analysis for the MindScope pipeline.

Responsibility (ONLY):
    Load the processed dataset and generate descriptive statistics
    and visualisations. Nothing in here modifies data or trains models.

Public API:
    perform_eda(df)  → runs all analyses and saves every figure/stat

All figures → reports/figures/
All stats   → reports/metrics/eda_stats.json

Input:  data/processed/processed_data.csv  (clean_text + label columns)
"""

import json
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from src.utils import config
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Consistent dark visual style across all plots ─────────────────────────
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

def perform_eda(df: pd.DataFrame) -> dict:
    """
    Run the full EDA suite on the processed dataset.

    Steps executed in order:
        1. Validate required columns exist.
        2. Compute basic stats (counts, class balance, imbalance ratio).
        3. Compute per-class word-count stats (mean, median, std).
        4. Plot class distribution bar chart.
        5. Plot text length KDE + box plots.
        6. Plot top-N words per class (horizontal bar charts).
        7. Plot word clouds per class.
        8. Save summary stats as JSON.

    Args:
        df (pd.DataFrame): Must contain `clean_text` and `label` columns.

    Returns:
        dict: Summary statistics dict (also written to reports/metrics/eda_stats.json).

    Example:
        >>> df = pd.read_csv("data/processed/processed_data.csv")
        >>> stats = perform_eda(df)
    """
    _validate_columns(df)
    ensure_dir(config.FIGURES_DIR)
    ensure_dir(config.METRICS_DIR)

    logger.info("=== EDA: Starting Exploratory Data Analysis ===")

    stats = _compute_basic_stats(df)

    df = df.copy()
    df["word_count"] = df[config.CLEAN_TEXT_COLUMN].str.split().str.len()

    stats["text_length_by_class"] = _compute_length_stats(df)

    _plot_class_distribution(df)
    _plot_text_length_dist(df)
    _plot_top_words(df)
    _plot_wordclouds(df)
    _save_stats(stats)

    logger.info("=== EDA: Complete. Figures → reports/figures/ ===")
    return stats


# ─────────────────────────────────────────────
# STATS HELPERS
# ─────────────────────────────────────────────

def _validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing."""
    required = {config.CLEAN_TEXT_COLUMN, config.LABEL_COLUMN}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"EDA requires columns {required}. Missing: {missing}\n"
            "Run run_pipeline.py first to generate processed_data.csv."
        )


def _compute_basic_stats(df: pd.DataFrame) -> dict:
    """
    Compute and log total samples, per-class counts/percentages,
    and imbalance ratio (max class / min class).

    Args:
        df (pd.DataFrame): Processed DataFrame with `label` column.

    Returns:
        dict: {'total_samples', 'class_counts', 'class_pct', 'imbalance_ratio'}
    """
    total  = len(df)
    counts = df[config.LABEL_COLUMN].value_counts().to_dict()
    pcts   = {k: round(100 * v / total, 2) for k, v in counts.items()}
    vals   = list(counts.values())
    ratio  = round(max(vals) / min(vals), 2) if min(vals) > 0 else float("inf")

    stats = {
        "total_samples":   total,
        "class_counts":    counts,
        "class_pct":       pcts,
        "imbalance_ratio": ratio,
        "num_classes":     len(counts),
    }

    logger.info(f"Total samples     : {total:,}")
    logger.info(f"Number of classes : {len(counts)}")
    logger.info(f"Imbalance ratio   : {ratio}x  (max_class / min_class)")
    for lbl in config.CLASSES:
        n = counts.get(lbl, 0)
        p = pcts.get(lbl, 0.0)
        logger.info(f"  {lbl:<12} {n:>6,} rows  ({p:.1f}%)")

    if ratio > 2.0:
        logger.warning(
            f"Significant class imbalance detected (ratio={ratio}). "
            "Consider class_weight='balanced' or oversampling."
        )
    return stats


def _compute_length_stats(df: pd.DataFrame) -> dict:
    """
    Compute word-count summary statistics (mean/median/std/min/max) per class.

    Args:
        df (pd.DataFrame): DataFrame with `word_count` and `label` columns.

    Returns:
        dict: {label: {mean, median, std, min, max}}
    """
    result = {}
    logger.info("Text length (word count) stats per class:")
    for lbl, grp in df.groupby(config.LABEL_COLUMN)["word_count"]:
        s = {
            "mean":   round(float(grp.mean()), 1),
            "median": round(float(grp.median()), 1),
            "std":    round(float(grp.std()), 1),
            "min":    int(grp.min()),
            "max":    int(grp.max()),
        }
        result[lbl] = s
        logger.info(
            f"  {lbl:<12} mean={s['mean']:>7}  median={s['median']:>7}"
            f"  std={s['std']:>7}  min={s['min']}  max={s['max']}"
        )
    return result


# ─────────────────────────────────────────────
# VISUALISATION FUNCTIONS
# ─────────────────────────────────────────────

def _plot_class_distribution(df: pd.DataFrame) -> None:
    """
    Save a horizontal bar chart of post counts per class.

    Each bar uses the class-specific colour from config.CLASS_COLORS.
    Bars are annotated with exact count and percentage.

    Output: reports/figures/class_distribution.png
    """
    counts = df[config.LABEL_COLUMN].value_counts()
    total  = len(df)
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = [config.CLASS_COLORS.get(l, "#AAAAAA") for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color=colors, height=0.55, zorder=2)

    for bar, val in zip(bars, values):
        pct = 100 * val / total
        ax.text(
            val + total * 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}  ({pct:.1f}%)",
            va="center", ha="left", fontsize=9, color="#C8CCDA",
        )

    ax.set_xlabel("Number of Posts")
    ax.set_title("Class Distribution — Reddit Mental Health Posts", pad=14)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="x", zorder=1)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    _save_figure(fig, "class_distribution.png")


def _plot_text_length_dist(df: pd.DataFrame) -> None:
    """
    Save a two-panel figure:
        Left  — Overlaid KDE curves of word count per class.
        Right — Box plots of word count per class.

    Output: reports/figures/text_length_dist.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Word Count Distribution by Class", fontsize=14, y=1.02)

    # KDE panel
    ax = axes[0]
    for lbl in config.CLASSES:
        subset = df[df[config.LABEL_COLUMN] == lbl]["word_count"]
        if subset.empty:
            continue
        sns.kdeplot(
            subset, ax=ax, label=lbl,
            color=config.CLASS_COLORS.get(lbl, "#AAAAAA"),
            linewidth=2, fill=True, alpha=0.15,
        )
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Density")
    ax.set_title("KDE per Class")
    ax.legend(framealpha=0.2)
    ax.set_xlim(left=0)

    # Box plot panel
    ax2 = axes[1]
    plot_data = [
        df[df[config.LABEL_COLUMN] == lbl]["word_count"].dropna().values
        for lbl in config.CLASSES
    ]
    bp = ax2.boxplot(
        plot_data,
        labels=config.CLASSES,
        patch_artist=True,
        medianprops=dict(color="#FFFFFF", linewidth=2),
        whiskerprops=dict(color="#C8CCDA"),
        capprops=dict(color="#C8CCDA"),
        flierprops=dict(
            marker="o", markerfacecolor="#C8CCDA",
            markersize=2, alpha=0.3, linestyle="none",
        ),
    )
    for patch, lbl in zip(bp["boxes"], config.CLASSES):
        patch.set_facecolor(config.CLASS_COLORS.get(lbl, "#AAAAAA"))
        patch.set_alpha(0.75)

    ax2.set_xlabel("Class")
    ax2.set_ylabel("Word Count")
    ax2.set_title("Box Plot per Class")
    ax2.grid(axis="y")

    fig.tight_layout()
    _save_figure(fig, "text_length_dist.png")


def _plot_top_words(df: pd.DataFrame) -> None:
    """
    Save a single figure with one horizontal bar chart per class,
    showing the top-N most frequent words.

    Output: reports/figures/top_words_per_class.png
    """
    n = len(config.CLASSES)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    fig.suptitle(f"Top {config.EDA_TOP_N_WORDS} Words per Class", fontsize=14, y=1.01)

    for ax, lbl in zip(axes, config.CLASSES):
        subset    = df[df[config.LABEL_COLUMN] == lbl][config.CLEAN_TEXT_COLUMN]
        word_freq = _get_top_words(subset, config.EDA_TOP_N_WORDS)
        color     = config.CLASS_COLORS.get(lbl, "#AAAAAA")

        if not word_freq:
            ax.set_title(lbl)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        words, freqs = zip(*word_freq)
        ax.barh(words, freqs, color=color, alpha=0.85)
        ax.invert_yaxis()
        ax.set_title(lbl.upper(), color=color, fontweight="bold", fontsize=11)
        ax.set_xlabel("Frequency")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x")
        ax.set_axisbelow(True)

    fig.tight_layout()
    _save_figure(fig, "top_words_per_class.png")


def _plot_wordclouds(df: pd.DataFrame) -> None:
    """
    Save one word cloud per class, arranged in a single 5-panel figure.

    Word size encodes relative frequency within each class corpus.
    Background and colour map match the dark theme.

    Output: reports/figures/wordclouds_per_class.png
    """
    n = len(config.CLASSES)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    fig.suptitle("Word Clouds per Class", fontsize=14, y=1.01)

    for ax, lbl in zip(axes, config.CLASSES):
        subset = df[df[config.LABEL_COLUMN] == lbl][config.CLEAN_TEXT_COLUMN]
        corpus = " ".join(subset.dropna().tolist())
        color  = config.CLASS_COLORS.get(lbl, "#AAAAAA")

        if not corpus.strip():
            ax.set_title(lbl)
            ax.axis("off")
            continue

        wc = WordCloud(
            width=config.WORDCLOUD_WIDTH,
            height=config.WORDCLOUD_HEIGHT,
            max_words=config.WORDCLOUD_MAX_WORDS,
            background_color="#1A1D27",
            colormap=_single_color_cmap(color),
            collocations=False,
            prefer_horizontal=0.85,
        ).generate(corpus)

        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(lbl.upper(), color=color, fontweight="bold", fontsize=11)
        ax.axis("off")

    fig.tight_layout()
    _save_figure(fig, "wordclouds_per_class.png")


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _get_top_words(series: pd.Series, n: int) -> list:
    """
    Return the n most frequent whitespace-separated tokens in a text Series.

    Args:
        series (pd.Series): Series of clean_text strings.
        n (int): Number of top words to return.

    Returns:
        list[tuple]: [(word, count), ...] sorted by descending count.
    """
    all_words = " ".join(series.dropna().tolist()).split()
    return Counter(all_words).most_common(n)


def _single_color_cmap(hex_color: str):
    """
    Build a two-stop colormap from near-black to the given hex colour.

    Gives word clouds a tinted look that matches the class identity colour.

    Args:
        hex_color (str): Hex string like "#E07B54".

    Returns:
        matplotlib.colors.LinearSegmentedColormap
    """
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    r, g, b = to_rgb(hex_color)
    return LinearSegmentedColormap.from_list(
        "cls_cmap", [(0.05, 0.05, 0.07), (r, g, b)]
    )


def _save_figure(fig: plt.Figure, filename: str) -> None:
    """
    Save a matplotlib Figure to reports/figures/ at 150 dpi, then close it.

    Args:
        fig (plt.Figure): Figure object to persist.
        filename (str):   Output filename, e.g. "class_distribution.png".
    """
    path = f"{config.FIGURES_DIR}/{filename}"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved figure: {path}")


def _save_stats(stats: dict) -> None:
    """
    Write the EDA summary statistics dict to JSON.

    Output: reports/metrics/eda_stats.json

    Args:
        stats (dict): Statistics dictionary from perform_eda().
    """
    path = f"{config.METRICS_DIR}/eda_stats.json"
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"EDA stats saved: {path}")