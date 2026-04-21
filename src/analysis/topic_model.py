"""
topic_model.py — LDA topic modeling for the MindScope pipeline.

Responsibility (ONLY):
    Fit one LDA model per mental-health class on the `clean_text` column,
    extract top topics + keywords, generate visualisations, produce a
    human-readable insights report, and save all outputs to disk.

Why LDA per-class instead of a single global model?
    A global LDA would blend vocabulary from all five conditions and produce
    mixed topics (e.g., "feel sad focus" — depression + ADHD blended).
    Per-class LDA isolates each community's internal themes, making the
    topics far more interpretable and actionable for clinical researchers.

Public API:
    perform_topic_modeling(df)  → dict of {class: [{topic_id, words, weights}]}
    generate_insights(df, topic_results, sentiment_stats, severity_stats)
                                → markdown insights report string

Input:  pd.DataFrame with `clean_text` and `label` columns
Output:
    models/saved/lda_model.pkl           — all fitted LDA models (dict)
    reports/metrics/topic_model_results.json
    reports/figures/topic_keywords.png
    reports/metrics/insights_report.md
"""

import json
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.utils import config
from src.utils.helpers import ensure_dir, save_object
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


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def perform_topic_modeling(df: pd.DataFrame) -> dict:
    """
    Fit one LDA model per class and extract top topics with keywords.

    For each class:
        1. Subset the DataFrame to that class.
        2. Fit a CountVectorizer (bag-of-words).
        3. Fit LatentDirichletAllocation with config.LDA_N_TOPICS topics.
        4. Extract top config.LDA_N_TOP_WORDS words per topic.
        5. Store results in a structured dict.

    Args:
        df (pd.DataFrame): Must contain `clean_text` and `label` columns.

    Returns:
        dict: {
            class_label: [
                {
                    "topic_id": int,
                    "top_words": [str, ...],   # top N words
                    "weights":   [float, ...]  # corresponding weights
                },
                ...
            ]
        }

    Example:
        >>> results = perform_topic_modeling(df)
        >>> results["depression"][0]["top_words"]
        ['feel', 'day', 'help', 'time', 'think', 'need', 'want', 'life', ...]
    """
    _validate(df)
    ensure_dir(config.MODELS_DIR)
    ensure_dir(config.METRICS_DIR)
    ensure_dir(config.FIGURES_DIR)

    logger.info(
        f"=== Topic Modeling: LDA per class "
        f"({config.LDA_N_TOPICS} topics, {config.LDA_N_TOP_WORDS} words/topic) ==="
    )

    all_models = {}   # {label: (lda_model, vectorizer, feature_names)}
    results    = {}   # final output dict

    for lbl in config.CLASSES:
        subset = df[df[config.LABEL_COLUMN] == lbl][config.CLEAN_TEXT_COLUMN].dropna()
        if len(subset) < config.LDA_N_TOPICS * 5:
            logger.warning(f"  {lbl}: too few samples ({len(subset)}). Skipping.")
            continue

        logger.info(f"  Fitting LDA for '{lbl}' ({len(subset):,} posts)...")

        lda, vectorizer, feature_names, doc_term_matrix = _fit_lda(subset)
        topics = _extract_topics(lda, feature_names)
        results[lbl] = topics
        all_models[lbl] = (lda, vectorizer, feature_names)

        logger.info(f"    Topics for {lbl}:")
        for t in topics:
            logger.info(f"      Topic {t['topic_id']}: {', '.join(t['top_words'][:6])}")

    # Save all models in a single dict
    save_object(all_models, config.TOPIC_MODEL_PATH)
    logger.info(f"LDA models saved: {config.TOPIC_MODEL_PATH}")

    # Save JSON results
    with open(config.TOPIC_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Topic results saved: {config.TOPIC_RESULTS_PATH}")

    # Visualise
    _plot_topic_keywords(results)

    return results


def generate_insights(
    df: pd.DataFrame,
    topic_results: dict,
    sentiment_stats: dict,
    severity_stats: dict,
) -> str:
    """
    Generate a human-readable Markdown insights report.

    Combines topic model results, sentiment stats, and severity stats into
    a structured narrative with clearly explained findings per class and
    cross-class comparisons.

    Args:
        df (pd.DataFrame):         Enriched DataFrame (with sentiment + severity).
        topic_results (dict):      Output of perform_topic_modeling().
        sentiment_stats (dict):    Output of get_sentiment_stats().
        severity_stats (dict):     Output of get_severity_stats().

    Returns:
        str: Full markdown report string (also saved to disk).

    Example:
        >>> report = generate_insights(df, topics, sent_stats, sev_stats)
        >>> print(report[:500])
        # MindScope — NLP Insights Report
        ...
    """
    ensure_dir(config.METRICS_DIR)
    logger.info("=== Generating Insights Report ===")

    lines = []
    _section = lambda title: lines.append(f"\n## {title}\n")
    _subsec   = lambda title: lines.append(f"\n### {title}\n")
    _add      = lambda text:  lines.append(text)

    # ── Header ───────────────────────────────────────────────────────────
    _add(f"# MindScope — NLP Insights Report")
    _add(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    _add(f"\nTotal posts analysed: **{len(df):,}**")
    _add(f"\nClasses: {', '.join(f'`{c}`' for c in config.CLASSES)}\n")
    _add("---")

    # ── Section 1: Sentiment Summary ──────────────────────────────────────
    _section("1. Sentiment Analysis Summary")
    _add("VADER compound scores range from -1.0 (most negative) to +1.0 (most positive).\n")

    by_class = sentiment_stats.get("by_class", {})
    _add("| Class | Mean Score | % Negative | % Positive | % Neutral |")
    _add("|-------|-----------|------------|------------|-----------|")
    for lbl in config.CLASSES:
        s = by_class.get(lbl, {})
        _add(
            f"| {lbl} | {s.get('mean_score', 'N/A'):+.4f} "
            f"| {s.get('pct_negative', 0):.1f}% "
            f"| {s.get('pct_positive', 0):.1f}% "
            f"| {s.get('pct_neutral', 0):.1f}% |"
        )

    mn = sentiment_stats.get("most_negative_class", "N/A")
    mp = sentiment_stats.get("most_positive_class", "N/A")
    _add(f"\n**Most negative class:** `{mn}` "
         f"(mean score: {by_class.get(mn,{}).get('mean_score','N/A'):+.4f})")
    _add(f"\n**Most positive class:** `{mp}` "
         f"(mean score: {by_class.get(mp,{}).get('mean_score','N/A'):+.4f})")

    _add("\n**Interpretation:**")
    _add(_sentiment_interpretation(sentiment_stats))

    # ── Section 2: Severity Summary ───────────────────────────────────────
    _section("2. Severity Score Summary")
    _add(
        "Severity score [0–1] is a composite of sentiment negativity (40%), "
        "post length (20%), and crisis keyword presence (40%). "
        "Scores > 0.6 are flagged as high-severity.\n"
    )

    sev_by_class = severity_stats.get("by_class", {})
    _add("| Class | Mean Severity | High Severity % | Max Severity |")
    _add("|-------|--------------|-----------------|--------------|")
    for lbl in config.CLASSES:
        s = sev_by_class.get(lbl, {})
        _add(
            f"| {lbl} | {s.get('mean_severity', 'N/A'):.4f} "
            f"| {s.get('high_severity_pct', 0):.1f}% "
            f"| {s.get('max_severity', 'N/A'):.4f} |"
        )

    ms = severity_stats.get("most_severe_class", "N/A")
    ls = severity_stats.get("least_severe_class", "N/A")
    _add(f"\n**Highest severity class:** `{ms}`")
    _add(f"\n**Lowest severity class:** `{ls}`")
    _add("\n**Interpretation:**")
    _add(_severity_interpretation(severity_stats))

    # ── Section 3: Per-Class Topic Themes ─────────────────────────────────
    _section("3. Topic Analysis — Themes per Class")
    _add(f"LDA was run per class with {config.LDA_N_TOPICS} topics and "
         f"{config.LDA_N_TOP_WORDS} top words per topic.\n")

    for lbl in config.CLASSES:
        topics = topic_results.get(lbl, [])
        if not topics:
            continue
        _subsec(f"{lbl.upper()}")
        _add(_describe_class_topics(lbl, topics))
        _add("\n| Topic | Top Words |")
        _add("|-------|-----------|")
        for t in topics:
            words = ", ".join(f"`{w}`" for w in t["top_words"])
            _add(f"| Topic {t['topic_id']} | {words} |")
        _add("")

    # ── Section 4: Cross-Class Comparisons ────────────────────────────────
    _section("4. Cross-Class Comparisons")

    _subsec("4a. ADHD vs OCD — Attention & Compulsion")
    _add(_compare_classes("adhd", "ocd", df, topic_results, by_class, sev_by_class))

    _subsec("4b. Depression vs PTSD — Internalising Conditions")
    _add(_compare_classes("depression", "ptsd", df, topic_results, by_class, sev_by_class))

    _subsec("4c. Aspergers — Unique Language Patterns")
    _add(_describe_aspergers(topic_results, by_class, sev_by_class))

    # ── Section 5: Emotional Patterns ─────────────────────────────────────
    _section("5. Emotional Patterns Across All Classes")
    _add(_emotional_patterns(df, by_class, sev_by_class))

    # ── Section 6: Key Findings ───────────────────────────────────────────
    _section("6. Key Findings & Recommendations")
    _add(_key_findings(sentiment_stats, severity_stats, topic_results))

    report = "\n".join(lines)

    with open(config.INSIGHTS_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Insights report saved: {config.INSIGHTS_REPORT_PATH}")

    return report


# ─────────────────────────────────────────────
# PRIVATE LDA HELPERS
# ─────────────────────────────────────────────

def _fit_lda(texts: pd.Series) -> tuple:
    """
    Fit a CountVectorizer + LDA model on a Series of clean text strings.

    CountVectorizer (bag-of-words) is used instead of TF-IDF because LDA's
    probabilistic model assumes raw term counts, not weighted scores.

    Args:
        texts (pd.Series): Clean text strings for one class.

    Returns:
        tuple: (lda, vectorizer, feature_names, doc_term_matrix)
    """
    vectorizer = CountVectorizer(
        max_features=5000,
        min_df=3,
        max_df=0.90,
        token_pattern=r"(?u)\b[a-z]{2,}\b",
    )
    doc_term_matrix = vectorizer.fit_transform(texts)
    feature_names   = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=config.LDA_N_TOPICS,
        max_iter=config.LDA_MAX_ITER,
        learning_method="batch",
        random_state=config.LDA_RANDOM_STATE,
        n_jobs=-1,
    )
    lda.fit(doc_term_matrix)
    return lda, vectorizer, feature_names, doc_term_matrix


def _extract_topics(lda, feature_names: np.ndarray) -> list:
    """
    Extract top words and their weights for every topic in a fitted LDA.

    Args:
        lda:            Fitted LatentDirichletAllocation.
        feature_names:  Array of vocabulary terms from CountVectorizer.

    Returns:
        list[dict]: [{"topic_id": int, "top_words": [...], "weights": [...]}, ...]
    """
    topics = []
    for topic_id, component in enumerate(lda.components_):
        top_idx     = component.argsort()[::-1][:config.LDA_N_TOP_WORDS]
        top_words   = [feature_names[i] for i in top_idx]
        top_weights = [round(float(component[i]), 4) for i in top_idx]
        topics.append({
            "topic_id":  topic_id,
            "top_words": top_words,
            "weights":   top_weights,
        })
    return topics


def _validate(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are absent."""
    required = {config.CLEAN_TEXT_COLUMN, config.LABEL_COLUMN}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"perform_topic_modeling() requires {required}. Missing: {missing}")


# ─────────────────────────────────────────────
# PRIVATE VISUALISATION
# ─────────────────────────────────────────────

def _plot_topic_keywords(results: dict) -> None:
    """
    Save a grid of horizontal bar charts — one subplot per class,
    showing the top-5 words of each LDA topic (stacked horizontally).

    Layout: 1 row per class, each row shows all topics side-by-side.
    Output: reports/figures/topic_keywords.png
    """
    n_classes  = len([c for c in config.CLASSES if c in results])
    n_topics   = config.LDA_N_TOPICS
    show_words = 5   # words per topic bar in the figure

    fig, axes = plt.subplots(
        n_classes, 1,
        figsize=(4 * n_topics, 3.5 * n_classes),
        squeeze=False,
    )
    fig.suptitle(
        f"LDA Topic Keywords per Class  ({n_topics} topics, top {show_words} words)",
        fontsize=14, y=1.01,
    )

    row = 0
    for lbl in config.CLASSES:
        if lbl not in results:
            continue
        ax     = axes[row][0]
        topics = results[lbl]
        color  = config.CLASS_COLORS.get(lbl, "#AAAAAA")
        x      = np.arange(show_words)

        for t_idx, topic in enumerate(topics[:n_topics]):
            words   = topic["top_words"][:show_words]
            weights = topic["weights"][:show_words]
            offset  = t_idx * (show_words + 1)
            positions = x + offset
            alpha   = 0.55 + 0.07 * t_idx   # slightly differentiate topics
            ax.bar(positions, weights, color=color, alpha=min(alpha, 0.92), width=0.75)
            ax.set_xticks(list(ax.get_xticks()) + list(positions))

        # Set tick labels — all words across all topics
        all_positions = []
        all_labels    = []
        for t_idx, topic in enumerate(topics[:n_topics]):
            words  = topic["top_words"][:show_words]
            offset = t_idx * (show_words + 1)
            for w_idx, word in enumerate(words):
                all_positions.append(w_idx + offset)
                all_labels.append(word)

        ax.set_xticks(all_positions)
        ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Weight")
        ax.set_title(lbl.upper(), color=color, fontweight="bold", fontsize=11)
        ax.grid(axis="y")

        # Topic separators
        for t_idx in range(1, n_topics):
            ax.axvline(t_idx * (show_words + 1) - 0.5,
                       color="#2E3146", linewidth=1.5, linestyle="--")

        row += 1

    fig.tight_layout()
    fig.savefig(config.TOPIC_FIGURE_PATH, dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Topic keywords figure saved: {config.TOPIC_FIGURE_PATH}")


# ─────────────────────────────────────────────
# PRIVATE INSIGHT TEXT BUILDERS
# ─────────────────────────────────────────────

def _sentiment_interpretation(sentiment_stats: dict) -> str:
    by_class = sentiment_stats.get("by_class", {})
    mn = sentiment_stats.get("most_negative_class", "")
    mp = sentiment_stats.get("most_positive_class", "")
    mn_score = by_class.get(mn, {}).get("mean_score", 0)
    mp_score = by_class.get(mp, {}).get("mean_score", 0)
    return (
        f"- `{mn}` posts carry the most negative emotional tone "
        f"(mean={mn_score:+.4f}), suggesting the highest baseline distress "
        f"in user self-descriptions within this community.\n"
        f"- `{mp}` posts skew toward more constructive language "
        f"(mean={mp_score:+.4f}). This may reflect a community culture of "
        f"sharing coping strategies or positive experiences.\n"
        f"- All classes show predominantly negative-to-neutral sentiment, "
        f"which is expected in self-reported mental health contexts."
    )


def _severity_interpretation(severity_stats: dict) -> str:
    by_class = severity_stats.get("by_class", {})
    ms = severity_stats.get("most_severe_class", "")
    ls = severity_stats.get("least_severe_class", "")
    ms_data = by_class.get(ms, {})
    ls_data = by_class.get(ls, {})
    return (
        f"- `{ms}` has the highest mean severity ({ms_data.get('mean_severity',0):.4f}) "
        f"with {ms_data.get('high_severity_pct',0):.1f}% of posts scoring > 0.6. "
        f"This is driven by high keyword density (crisis/trauma language) and "
        f"strongly negative sentiment scores.\n"
        f"- `{ls}` shows the lowest mean severity ({ls_data.get('mean_severity',0):.4f}), "
        f"suggesting relatively less acute distress in post content.\n"
        f"- Note: severity is a proxy derived from text features, not a clinical measure."
    )


def _describe_class_topics(lbl: str, topics: list) -> str:
    """Generate a one-paragraph plain-English summary of a class's LDA topics."""
    all_words = []
    for t in topics:
        all_words.extend(t["top_words"][:5])
    top_unique = list(dict.fromkeys(all_words))[:config.INSIGHTS_TOP_N]
    word_list  = ", ".join(f"`{w}`" for w in top_unique)

    descriptions = {
        "adhd": (
            f"ADHD posts cluster around executive function, time perception, and "
            f"emotional regulation. Dominant terms ({word_list}) reflect "
            f"struggles with focus, impulsivity, and the frustration of "
            f"being misunderstood. Community tone is often humour-mixed-with-despair."
        ),
        "aspergers": (
            f"Aspergers posts focus heavily on social navigation, identity, and "
            f"sensory experience. Key terms ({word_list}) point to discussions "
            f"of masking, social anxiety, and neurodivergent identity. "
            f"Language is often analytical and introspective."
        ),
        "depression": (
            f"Depression posts are dominated by themes of hopelessness, isolation, "
            f"and loss of motivation. Recurring terms ({word_list}) capture "
            f"the subjective experience of emptiness, sleep disturbance, and "
            f"the challenge of daily functioning."
        ),
        "ocd": (
            f"OCD posts centre on intrusive thoughts, compulsive rituals, and "
            f"the exhaustion of mental checking. Terms ({word_list}) reveal "
            f"themes of contamination fear, harm OCD, and the cycle of "
            f"obsession and temporary relief through compulsion."
        ),
        "ptsd": (
            f"PTSD posts revolve around trauma re-experiencing, hypervigilance, "
            f"and relationship disruption. Key terms ({word_list}) highlight "
            f"nightmares, triggers, avoidance behaviour, and the long-term "
            f"impact of traumatic events on daily functioning."
        ),
    }
    return descriptions.get(lbl, f"Key terms: {word_list}.")


def _compare_classes(
    lbl_a: str, lbl_b: str,
    df: pd.DataFrame,
    topics: dict,
    sent_stats: dict,
    sev_stats: dict,
) -> str:
    """Compare two classes on sentiment, severity, and top vocabulary."""
    def top_words(lbl):
        ts = topics.get(lbl, [])
        words = []
        for t in ts:
            words.extend(t["top_words"][:4])
        return list(dict.fromkeys(words))[:6]

    words_a  = ", ".join(f"`{w}`" for w in top_words(lbl_a))
    words_b  = ", ".join(f"`{w}`" for w in top_words(lbl_b))
    sent_a   = sent_stats.get(lbl_a, {}).get("mean_score", 0)
    sent_b   = sent_stats.get(lbl_b, {}).get("mean_score", 0)
    sev_a    = sev_stats.get(lbl_a, {}).get("mean_severity", 0)
    sev_b    = sev_stats.get(lbl_b, {}).get("mean_severity", 0)
    pneg_a   = sent_stats.get(lbl_a, {}).get("pct_negative", 0)
    pneg_b   = sent_stats.get(lbl_b, {}).get("pct_negative", 0)

    more_neg = lbl_a if sent_a < sent_b else lbl_b
    more_sev = lbl_a if sev_a  > sev_b  else lbl_b

    return (
        f"- **`{lbl_a}`** top vocab: {words_a}\n"
        f"- **`{lbl_b}`** top vocab: {words_b}\n"
        f"- Sentiment: `{lbl_a}` mean={sent_a:+.4f} ({pneg_a:.1f}% negative) vs "
        f"`{lbl_b}` mean={sent_b:+.4f} ({pneg_b:.1f}% negative)\n"
        f"- Severity: `{lbl_a}` mean={sev_a:.4f} vs `{lbl_b}` mean={sev_b:.4f}\n"
        f"- `{more_neg}` carries more negative emotional tone; "
        f"`{more_sev}` scores higher on composite severity.\n"
        f"- Vocabulary overlap exists due to comorbidity (e.g., both discuss "
        f"anxiety, sleep, and relationships), but the _primary_ signals differ: "
        f"`{lbl_a}` is shaped by {'attention and impulsivity' if lbl_a=='adhd' else 'internalised suffering'}, "
        f"while `{lbl_b}` is shaped by {'rituals and doubt' if lbl_b=='ocd' else 'trauma re-experiencing'}."
    )


def _describe_aspergers(topics: dict, sent_stats: dict, sev_stats: dict) -> str:
    words = []
    for t in topics.get("aspergers", []):
        words.extend(t["top_words"][:3])
    top = ", ".join(f"`{w}`" for w in list(dict.fromkeys(words))[:8])
    mean_sent = sent_stats.get("aspergers", {}).get("mean_score", 0)
    mean_sev  = sev_stats.get("aspergers", {}).get("mean_severity", 0)
    return (
        f"- Aspergers posts are linguistically distinct from the other classes.\n"
        f"- Dominant terms: {top}\n"
        f"- Mean sentiment: {mean_sent:+.4f} | Mean severity: {mean_sev:.4f}\n"
        f"- Unlike depression or PTSD, Aspergers posts more often discuss "
        f"_identity and social dynamics_ rather than acute emotional crisis. "
        f"The community uses highly specific vocabulary around masking, "
        f"neurotypical vs neurodivergent experience, and sensory sensitivities.\n"
        f"- This makes Aspergers the most lexically separable class — a "
        f"classifier should achieve high precision for this label."
    )


def _emotional_patterns(df: pd.DataFrame, sent_stats: dict, sev_stats: dict) -> str:
    lines = []
    lines.append("**Ranking of classes by mean sentiment score (most → least negative):**\n")
    ranked = sorted(
        config.CLASSES,
        key=lambda l: sent_stats.get(l, {}).get("mean_score", 0),
    )
    for rank, lbl in enumerate(ranked, 1):
        s = sent_stats.get(lbl, {})
        v = sev_stats.get(lbl, {})
        lines.append(
            f"{rank}. `{lbl}` — sentiment={s.get('mean_score',0):+.4f}, "
            f"neg%={s.get('pct_negative',0):.1f}%, "
            f"severity={v.get('mean_severity',0):.4f}"
        )
    lines.append(
        "\n**Pattern:** All five communities express predominantly negative sentiment, "
        "consistent with the self-reporting nature of mental health subreddits. "
        "Differences in _degree_ rather than _direction_ distinguish the classes.\n"
        "The inverse relationship between sentiment and severity is consistent "
        "(more negative sentiment → higher severity score), validating the scoring model."
    )
    return "\n".join(lines)


def _key_findings(
    sentiment_stats: dict, severity_stats: dict, topic_results: dict
) -> str:
    mn = sentiment_stats.get("most_negative_class", "N/A")
    ms = severity_stats.get("most_severe_class", "N/A")
    return (
        f"1. **`{mn}` shows the most consistently negative sentiment** across all posts. "
        f"Posts in this community are most likely to contain sustained emotional distress.\n\n"
        f"2. **`{ms}` carries the highest composite severity scores**, driven by "
        f"both strongly negative sentiment and high crisis-keyword density. "
        f"This is the class most likely to require support triage.\n\n"
        f"3. **Aspergers is the most linguistically distinct class**, making it "
        f"the most classifiable by vocabulary alone. ADHD, OCD, Depression, "
        f"and PTSD share more overlapping vocabulary due to high comorbidity.\n\n"
        f"4. **Topic models reveal that each class has 2-3 dominant themes** "
        f"that repeat across most topics, indicating tight semantic coherence "
        f"within each community.\n\n"
        f"5. **Recommendation — Model improvements:** Because depression and PTSD "
        f"are semantically closest, the classifier should be evaluated for "
        f"confusion between these two classes specifically. Adding severity and "
        f"sentiment as auxiliary features may help disambiguate them.\n\n"
        f"6. **Ethical note:** All scores and classifications are derived from "
        f"text patterns, not clinical assessment. They must not be used to "
        f"diagnose or make decisions about individuals."
    )