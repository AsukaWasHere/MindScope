"""
app.py — MindScope Interactive Dashboard

Streamlit application that provides a polished, professional interface for:
  • Text-based mental health condition classification
  • Real-time sentiment analysis (VADER)
  • Composite severity scoring
  • Per-class confidence visualisation
  • Project-level dataset insights panel

Run with:
    streamlit run app.py

Architecture:
    This app is a pure UI layer. All inference logic lives in:
        src/modeling/predictor.py   — prediction pipeline
        src/analysis/sentiment_analyzer.py — VADER scoring
        src/analysis/severity_scorer.py    — severity computation
    The app calls those modules directly — no logic is duplicated here.
"""

import os
import sys
import json
import time

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Ensure project root is on sys.path so `src.*` imports work ─────────────
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import config

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="MindScope — Mental Health NLP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — refined dark clinical aesthetic
# ─────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@300;400;500&display=swap');

  /* ── Base ─────────────────────────────── */
  html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0C0E16;
    color: #C8CCDA;
  }

  /* ── Header ───────────────────────────── */
  .ms-hero {
    background: linear-gradient(135deg, #131627 0%, #0C0E16 60%);
    border-bottom: 1px solid #1E2235;
    padding: 2.5rem 2rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
  }
  .ms-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #C8CCDA 0%, #7A6BAE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
  }
  .ms-subtitle {
    color: #5B6280;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.4rem;
  }

  /* ── Cards ────────────────────────────── */
  .ms-card {
    background: #131627;
    border: 1px solid #1E2235;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1rem;
  }
  .ms-card-accent {
    border-left: 3px solid var(--accent, #7A6BAE);
  }
  .ms-section-label {
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5B6280;
    margin-bottom: 0.6rem;
  }

  /* ── Prediction badge ─────────────────── */
  .ms-badge {
    display: inline-block;
    padding: 0.45rem 1.1rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  /* ── Condition display ────────────────── */
  .ms-condition {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    line-height: 1;
    margin: 0.3rem 0 0.6rem;
  }

  /* ── Metric row ───────────────────────── */
  .ms-metric-row {
    display: flex;
    gap: 1rem;
    margin: 0.8rem 0;
  }
  .ms-metric {
    flex: 1;
    background: #0C0E16;
    border: 1px solid #1E2235;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    text-align: center;
  }
  .ms-metric-val {
    font-size: 1.6rem;
    font-weight: 500;
    line-height: 1;
    margin-bottom: 0.2rem;
  }
  .ms-metric-label {
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #5B6280;
  }

  /* ── Progress bar ─────────────────────── */
  .ms-bar-track {
    background: #1E2235;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin: 0.4rem 0 0.2rem;
  }
  .ms-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
  }

  /* ── Warning banner ───────────────────── */
  .ms-warn {
    background: #1A1308;
    border: 1px solid #3D2E10;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.78rem;
    color: #C8A96E;
    margin-top: 0.8rem;
  }

  /* ── Sidebar ──────────────────────────── */
  [data-testid="stSidebar"] {
    background: #0E1020;
    border-right: 1px solid #1E2235;
  }
  .ms-sidebar-head {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #C8CCDA;
    margin-bottom: 0.2rem;
  }

  /* ── Streamlit overrides ──────────────── */
  .stTextArea textarea {
    background: #131627 !important;
    border: 1px solid #2A2E45 !important;
    border-radius: 8px !important;
    color: #C8CCDA !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
  }
  .stTextArea textarea:focus {
    border-color: #7A6BAE !important;
    box-shadow: 0 0 0 2px rgba(122,107,174,0.15) !important;
  }
  .stButton > button {
    background: linear-gradient(135deg, #7A6BAE 0%, #5B8DB8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.55rem 1.8rem !important;
    font-weight: 500 !important;
    transition: opacity 0.2s !important;
  }
  .stButton > button:hover { opacity: 0.85 !important; }
  div[data-testid="stHorizontalBlock"] > div { gap: 1rem; }
  .element-container { margin-bottom: 0 !important; }

  /* ── Tabs ─────────────────────────────── */
  .stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS & METADATA
# ─────────────────────────────────────────────

# Human-readable names and descriptions for each class
CLASS_META = {
    "adhd": {
        "name":  "ADHD",
        "full":  "Attention Deficit Hyperactivity Disorder",
        "desc":  "Difficulty sustaining attention, impulsivity, executive dysfunction.",
        "emoji": "⚡",
        "color": "#E07B54",
    },
    "aspergers": {
        "name":  "Asperger's",
        "full":  "Autism Spectrum (Asperger's)",
        "desc":  "Social navigation difficulty, sensory sensitivity, neurodivergent identity.",
        "emoji": "🔷",
        "color": "#5B8DB8",
    },
    "depression": {
        "name":  "Depression",
        "full":  "Major Depressive Disorder",
        "desc":  "Persistent low mood, hopelessness, anhedonia, fatigue.",
        "emoji": "🌧",
        "color": "#7A6BAE",
    },
    "ocd": {
        "name":  "OCD",
        "full":  "Obsessive-Compulsive Disorder",
        "desc":  "Intrusive thoughts, compulsive rituals, mental checking loops.",
        "emoji": "🔁",
        "color": "#4FAF8C",
    },
    "ptsd": {
        "name":  "PTSD",
        "full":  "Post-Traumatic Stress Disorder",
        "desc":  "Trauma re-experiencing, hypervigilance, avoidance, flashbacks.",
        "emoji": "🌊",
        "color": "#C0575A",
    },
}

SENTIMENT_COLORS = {
    "positive": "#4FAF8C",
    "neutral":  "#5B8DB8",
    "negative": "#C0575A",
}

SEVERITY_COLOR_STOPS = [
    (0.0,  "#4FAF8C"),   # low   — green
    (0.4,  "#E0C050"),   # mid   — amber
    (0.6,  "#E07B54"),   # high  — orange
    (0.85, "#C0575A"),   # very high — red
]


# ─────────────────────────────────────────────
# CACHED RESOURCE LOADERS
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_predictor():
    """
    Load the full inference stack once and cache across Streamlit reruns.

    Returns the `predict` function from predictor.py. Because predict()
    uses module-level caching internally, artifacts are only read from
    disk once per process lifetime.

    Returns:
        callable: The predict(raw_text) function.

    Raises:
        FileNotFoundError: If models/saved/best_model.pkl is missing.
                           User must run run_pipeline.py first.
    """
    from src.modeling.predictor import predict
    return predict


@st.cache_resource(show_spinner=False)
def load_sentiment_analyzer():
    """
    Load VADER SentimentIntensityAnalyzer once, cached across reruns.

    Returns:
        callable: Function (text: str) → dict with sentiment_score and sentiment_label.
    """
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    def analyze(text: str) -> dict:
        score = sia.polarity_scores(text)["compound"]
        if score >= config.VADER_POSITIVE_THRESH:
            label = "positive"
        elif score <= config.VADER_NEGATIVE_THRESH:
            label = "negative"
        else:
            label = "neutral"
        return {"sentiment_score": round(score, 4), "sentiment_label": label}

    return analyze


@st.cache_data(show_spinner=False)
def load_eda_stats() -> dict:
    """
    Load pre-computed EDA stats JSON if it exists.

    Returns:
        dict: EDA statistics, or empty dict if file not found.
    """
    if os.path.exists(config.SENTIMENT_STATS_PATH):
        with open(config.SENTIMENT_STATS_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_severity_stats() -> dict:
    if os.path.exists(config.SEVERITY_STATS_PATH):
        with open(config.SEVERITY_STATS_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame | None:
    if os.path.exists(config.COMPARISON_TABLE_PATH):
        return pd.read_csv(config.COMPARISON_TABLE_PATH)
    return None


# ─────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────

def run_severity(clean_text: str, sentiment_score: float) -> float:
    """
    Compute severity score using the same formula as severity_scorer.py.

    Reuses config constants directly — no code duplication.
    This function mirrors _sentiment_to_severity + _keyword_score + length
    weighting from severity_scorer.py but operates on already-preprocessed
    text (clean_text) rather than raw text, which is correct for the app.

    Args:
        clean_text (str):       Preprocessed text string.
        sentiment_score (float): VADER compound score [-1, +1].

    Returns:
        float: Severity score in [0.0, 1.0].
    """
    # Signal 1 — sentiment negativity
    S_sent = max(0.0, min(1.0, (1.0 - sentiment_score) / 2.0))

    # Signal 2 — post length (normalised)
    word_count = len(clean_text.split())
    S_len = min(1.0, word_count / config.SEVERITY_LENGTH_CAP)

    # Signal 3 — crisis keyword presence
    lowered = clean_text.lower()
    matched = sum(1 for kw in config.SEVERITY_KEYWORDS if kw in lowered)
    S_kw    = min(1.0, matched / max(1, len(config.SEVERITY_KEYWORDS)))

    raw = (
        config.SEVERITY_SENTIMENT_WEIGHT * S_sent
        + config.SEVERITY_LENGTH_WEIGHT  * S_len
        + config.SEVERITY_KEYWORD_WEIGHT * S_kw
    )
    return round(float(np.clip(raw, 0.0, 1.0)), 4)


def severity_color(score: float) -> str:
    """Return a hex colour interpolated across SEVERITY_COLOR_STOPS."""
    for i in range(len(SEVERITY_COLOR_STOPS) - 1):
        t0, c0 = SEVERITY_COLOR_STOPS[i]
        t1, c1 = SEVERITY_COLOR_STOPS[i + 1]
        if score <= t1:
            frac = (score - t0) / max(t1 - t0, 1e-9)
            # Linear interpolation in RGB
            r0, g0, b0 = int(c0[1:3], 16), int(c0[3:5], 16), int(c0[5:7], 16)
            r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
            r = int(r0 + frac * (r1 - r0))
            g = int(g0 + frac * (g1 - g0))
            b = int(b0 + frac * (b1 - b0))
            return f"#{r:02x}{g:02x}{b:02x}"
    return SEVERITY_COLOR_STOPS[-1][1]


def normalize_scores_to_pct(scores: dict) -> dict:
    """
    Convert raw decision-function scores (LinearSVC) to a 0-100 percentage
    scale via softmax-like normalisation so the chart is always readable.

    For models with true probabilities, values are already in [0,1].
    """
    vals = list(scores.values())
    keys = list(scores.keys())
    # Shift so minimum is 0, then normalise
    shifted = [v - min(vals) for v in vals]
    total   = sum(shifted) or 1.0
    normed  = {k: round(100 * v / total, 1) for k, v in zip(keys, shifted)}
    return normed


# ─────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────

def build_confidence_chart(scores: dict, predicted: str) -> go.Figure:
    """
    Horizontal bar chart of per-class confidence / decision scores.

    Predicted class bar is highlighted with full opacity; others are dimmed.
    Uses the project's CLASS_COLORS palette for consistency.

    Args:
        scores (dict):    {class: score} from predictor.predict().
        predicted (str):  The winning class label.

    Returns:
        plotly Figure.
    """
    pct = normalize_scores_to_pct(scores)
    sorted_items = sorted(pct.items(), key=lambda x: x[1], reverse=True)
    labels = [CLASS_META.get(k, {}).get("name", k) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    keys   = [k for k, _ in sorted_items]
    
    def hex_to_rgba(hex_color, alpha=1.0):
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    
    colors = [
    hex_to_rgba(
        config.CLASS_COLORS.get(k, "#AAAAAA"),
        1.0 if k == predicted else 0.3
    )
    for k in keys
]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=11, color="#C8CCDA"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=0, r=40, t=10, b=0),
        height=220,
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, max(values) * 1.25]),
        yaxis=dict(showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#C8CCDA", size=11),
    )
    return fig


def build_sentiment_gauge(score: float) -> go.Figure:
    """
    Semi-circular gauge showing the VADER compound sentiment score.

    Range: -1.0 (most negative) to +1.0 (most positive).
    Zones: red (negative), grey (neutral), green (positive).

    Args:
        score (float): VADER compound score in [-1.0, +1.0].

    Returns:
        plotly Figure.
    """
    # Map score from [-1, +1] → [0, 1] for gauge display
    norm = (score + 1) / 2

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(
            font=dict(size=28, family="DM Mono, monospace", color="#C8CCDA"),
            suffix="",
        ),
        gauge=dict(
            axis=dict(
                range=[-1, 1],
                tickwidth=1,
                tickcolor="#2E3146",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.0", "-0.5", "0", "+0.5", "+1.0"],
                tickfont=dict(size=9, color="#5B6280"),
            ),
            bar=dict(color=SENTIMENT_COLORS.get(
                "negative" if score < -0.05 else "positive" if score > 0.05 else "neutral",
                "#5B8DB8"
            ), thickness=0.3),
            bgcolor="#1E2235",
            borderwidth=0,
            steps=[
                dict(range=[-1.0, -0.05], color="#2A1820"),
                dict(range=[-0.05, 0.05],  color="#1A1D27"),
                dict(range=[0.05,  1.0],   color="#18271E"),
            ],
            threshold=dict(
                line=dict(color="#C8CCDA", width=2),
                thickness=0.7,
                value=score,
            ),
        ),
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#C8CCDA"),
    )
    return fig


def build_dataset_overview_chart(sentiment_stats: dict, severity_stats: dict) -> go.Figure:
    """
    Grouped bar chart showing mean sentiment and mean severity per class
    across the training dataset — gives context for how the user's post
    compares to the community baseline.

    Args:
        sentiment_stats (dict): Output of get_sentiment_stats().
        severity_stats (dict):  Output of get_severity_stats().

    Returns:
        plotly Figure.
    """
    classes     = config.CLASSES
    sent_means  = [sentiment_stats.get("by_class", {}).get(c, {}).get("mean_score", 0)
                   for c in classes]
    sev_means   = [severity_stats.get("by_class", {}).get(c, {}).get("mean_severity", 0)
                   for c in classes]
    class_names = [CLASS_META[c]["name"] for c in classes]
    colors      = [config.CLASS_COLORS[c] for c in classes]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Mean Sentiment Score", "Mean Severity Score"),
                        horizontal_spacing=0.12)

    for i, (cls, name, color, sent, sev) in enumerate(
        zip(classes, class_names, colors, sent_means, sev_means)
    ):
        fig.add_trace(go.Bar(
            name=name, x=[name], y=[sent], marker_color=color,
            marker_opacity=0.8,
            showlegend=False,
            text=[f"{sent:+.3f}"],
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate=f"{name}: {{y:.4f}}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            name=name, x=[name], y=[sev], marker_color=color,
            marker_opacity=0.8,
            showlegend=False,
            text=[f"{sev:.3f}"],
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate=f"{name}: {{y:.4f}}<extra></extra>",
        ), row=1, col=2)

    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#C8CCDA", size=10),
        bargap=0.3,
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=9))
    fig.update_yaxes(showgrid=True, gridcolor="#1E2235", zeroline=True,
                     zerolinecolor="#2E3146")
    return fig


# ─────────────────────────────────────────────
# UI COMPONENT BUILDERS
# ─────────────────────────────────────────────

def render_hero():
    """Render the top header banner."""
    st.markdown("""
    <div class="ms-hero">
      <p class="ms-title">🧠 MindScope</p>
      <p class="ms-subtitle">Mental Health NLP Analysis System &nbsp;·&nbsp; Research Tool</p>
    </div>
    """, unsafe_allow_html=True)


def render_ethical_notice():
    """Render the persistent ethical disclaimer."""
    st.markdown("""
    <div class="ms-warn">
      ⚠️ <strong>Research tool only.</strong> Predictions are derived from
      statistical text patterns, not clinical assessment. This system must
      never be used to diagnose, label, or make decisions about real individuals.
    </div>
    """, unsafe_allow_html=True)


def render_condition_card(predicted: str, confidence: float | None):
    """
    Render the primary prediction result card with condition name,
    full description, and confidence badge.

    Args:
        predicted (str):          Class key e.g. "depression".
        confidence (float|None):  Probability [0,1] or None for SVM.
    """
    meta  = CLASS_META.get(predicted, {})
    color = meta.get("color", "#7A6BAE")
    name  = meta.get("name", predicted.title())
    full  = meta.get("full", "")
    desc  = meta.get("desc", "")
    emoji = meta.get("emoji", "●")

    conf_html = ""
    if confidence is not None:
        conf_html = (
            f'<span class="ms-badge" '
            f'style="background:{color}22; color:{color}; border:1px solid {color}55;">'
            f'{confidence*100:.1f}% confidence</span>'
        )

    st.markdown(f"""
    <div class="ms-card ms-card-accent" style="--accent:{color};">
      <p class="ms-section-label">Detected Condition</p>
      <p class="ms-condition" style="color:{color};">{emoji}&nbsp; {name}</p>
      <p style="color:#8890AA; font-size:0.8rem; margin:0 0 0.6rem;">{full}</p>
      <p style="color:#C8CCDA; font-size:0.85rem; margin:0 0 0.8rem;">{desc}</p>
      {conf_html}
    </div>
    """, unsafe_allow_html=True)


def render_sentiment_card(sentiment_score: float, sentiment_label: str):
    """
    Render the sentiment result card with score, label badge, and gauge.

    Args:
        sentiment_score (float):  VADER compound score [-1.0, +1.0].
        sentiment_label (str):    "positive" / "neutral" / "negative".
    """
    color = SENTIMENT_COLORS.get(sentiment_label, "#5B8DB8")
    label_upper = sentiment_label.upper()

    st.markdown(f"""
    <div class="ms-card ms-card-accent" style="--accent:{color};">
      <p class="ms-section-label">Sentiment Analysis — VADER</p>
      <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.4rem;">
        <span class="ms-badge"
          style="background:{color}22; color:{color}; border:1px solid {color}55;">
          {label_upper}
        </span>
        <span style="font-size:1.4rem; font-weight:500; color:{color};">
          {sentiment_score:+.4f}
        </span>
      </div>
      <p style="color:#5B6280; font-size:0.72rem; margin:0;">
        Compound score range: −1.0 (most negative) → +1.0 (most positive)
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(build_sentiment_gauge(sentiment_score),
                    use_container_width=True, config={"displayModeBar": False})


def render_severity_card(severity: float):
    """
    Render the severity score card with value, colour-coded bar, and
    plain-English interpretation of the score band.

    Args:
        severity (float): Severity score in [0.0, 1.0].
    """
    color = severity_color(severity)
    pct   = int(severity * 100)

    if severity < 0.3:
        level, interp = "Low", "Post language shows relatively mild distress signals."
    elif severity < 0.5:
        level, interp = "Moderate", "Some distress indicators present in the text."
    elif severity < 0.7:
        level, interp = "Elevated", "Significant distress language detected."
    else:
        level, interp = "High", "Strong distress and crisis language present."

    st.markdown(f"""
    <div class="ms-card ms-card-accent" style="--accent:{color};">
      <p class="ms-section-label">Composite Severity Score</p>
      <div style="display:flex; align-items:baseline; gap:0.8rem; margin-bottom:0.5rem;">
        <span style="font-size:2.5rem; font-weight:500; color:{color}; line-height:1;">
          {severity:.4f}
        </span>
        <span class="ms-badge"
          style="background:{color}22; color:{color}; border:1px solid {color}55;">
          {level}
        </span>
      </div>
      <div class="ms-bar-track">
        <div class="ms-bar-fill"
             style="width:{pct}%; background:linear-gradient(90deg, #4FAF8C, {color});">
        </div>
      </div>
      <p style="display:flex; justify-content:space-between; color:#5B6280;
                font-size:0.68rem; margin:0.15rem 0 0.6rem;">
        <span>0.0 — minimal</span><span>1.0 — severe</span>
      </p>
      <p style="color:#8890AA; font-size:0.78rem; margin:0;">{interp}</p>
      <p style="color:#3D4258; font-size:0.68rem; margin:0.4rem 0 0;">
        Composite: 40% sentiment · 20% length · 40% crisis keywords
      </p>
    </div>
    """, unsafe_allow_html=True)


def render_processing_detail(raw_text: str, clean_text: str, word_count: int):
    """
    Render a collapsible section showing the preprocessing pipeline output.

    Args:
        raw_text (str):   Original user input.
        clean_text (str): Tokenised, lemmatised, stopword-removed version.
        word_count (int): Token count of clean_text.
    """
    with st.expander("🔬 Preprocessing Pipeline Detail", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Raw Input**")
            st.code(raw_text[:500] + ("..." if len(raw_text) > 500 else ""),
                    language=None)
        with col2:
            st.markdown("**After Preprocessing**")
            st.code(clean_text[:500] + ("..." if len(clean_text) > 500 else ""),
                    language=None)
        st.caption(f"Token count after preprocessing: **{word_count}** words")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar():
    """
    Render the sidebar with project info, example posts, and
    a brief legend for each mental health class.
    """
    with st.sidebar:
        st.markdown('<p class="ms-sidebar-head">🧠 MindScope</p>', unsafe_allow_html=True)
        st.caption("Mental Health NLP Research Dashboard")
        st.divider()

        # ── Example Prompts ────────────────────────────────────────────────
        st.markdown("**📝 Example Posts**")
        st.caption("Click to load into the analyser")

        examples = {
            "Depression": "I haven't left my bed in three days. Everything feels pointless and I can't remember the last time I felt anything other than empty.",
            "ADHD": "I forgot my keys again, missed two meetings, and spent 4 hours hyperfocused on a random Wikipedia rabbit hole instead of finishing my report.",
            "OCD": "I've checked the stove seventeen times tonight and I still can't shake the feeling it might be on. The thought won't leave me alone.",
            "PTSD": "The nightmare was back again. I woke up screaming and couldn't remember where I was. Every loud noise today sent me into a spiral.",
            "Asperger's": "I've been studying social cues for years but I still can't tell when someone wants me to stop talking. Masking is exhausting.",
        }

        for label, text in examples.items():
            color = config.CLASS_COLORS.get(label.lower().replace("'s","ers").replace(" ",""), "#AAAAAA")
            if st.button(
                f"{CLASS_META.get(label.lower().replace(chr(39)+'s','ers').replace(' ',''), {}).get('emoji','●')} {label}",
                key=f"ex_{label}",
                use_container_width=True,
            ):
                st.session_state["example_text"] = text

        st.divider()

        # ── Class Legend ───────────────────────────────────────────────────
        st.markdown("**📚 Condition Classes**")
        for key, meta in CLASS_META.items():
            color = meta["color"]
            st.markdown(
                f'<div style="display:flex; align-items:center; gap:0.5rem; '
                f'margin-bottom:0.4rem;">'
                f'<span style="width:8px; height:8px; border-radius:50%; '
                f'background:{color}; flex-shrink:0; display:inline-block;"></span>'
                f'<span style="font-size:0.75rem; color:#C8CCDA;">{meta["name"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption("Built with MindScope NLP Pipeline")
        st.caption(f"Classes: {', '.join(config.CLASSES)}")


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    """
    Main Streamlit application entry point.

    Layout:
        Sidebar   — examples, class legend, project info
        Main area — two tabs:
            Tab 1: Analyser     — text input + full prediction output
            Tab 2: Dataset View — pre-computed insights from training data
    """
    render_sidebar()
    render_hero()

    # ── Load backend resources (cached) ────────────────────────────────────
    artifacts_ready = True
    try:
        predict_fn       = load_predictor()
        analyze_sent_fn  = load_sentiment_analyzer()
    except Exception as e:
        st.error(
            f"⚠️ **Model artifacts not found.** Run the full pipeline first:\n\n"
            f"```bash\npython run_pipeline.py\n```\n\n"
            f"Error: `{e}`"
        )
        artifacts_ready = False

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab_analyse, tab_dataset = st.tabs(["🔍  Analyse Text", "📊  Dataset Insights"])

    # ══════════════════════════════════════════
    # TAB 1 — ANALYSER
    # ══════════════════════════════════════════
    with tab_analyse:
        render_ethical_notice()
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Input area ────────────────────────────────────────────────────
        default_text = st.session_state.pop("example_text", "")
        col_input, col_meta = st.columns([2, 1])

        with col_input:
            st.markdown(
                '<p class="ms-section-label">Post Text</p>',
                unsafe_allow_html=True,
            )
            user_input = st.text_area(
                label="",
                value=default_text,
                height=180,
                placeholder=(
                    "Paste or type a Reddit post here.\n\n"
                    "Example: 'I haven't been able to sleep for weeks, "
                    "the nightmares keep coming back every night...'"
                ),
                key="user_text_input",
                label_visibility="collapsed",
            )

        with col_meta:
            st.markdown(
                '<p class="ms-section-label">Analysis Options</p>',
                unsafe_allow_html=True,
            )
            show_preprocessing = st.toggle("Show preprocessing detail", value=False)
            show_confidence    = st.toggle("Show confidence chart", value=True)
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("▶  Analyse Text", use_container_width=True,
                                disabled=not artifacts_ready)

        # ── Run inference ──────────────────────────────────────────────────
        if run_btn:
            _run_analysis(
                user_input, predict_fn, analyze_sent_fn,
                show_preprocessing, show_confidence,
            )

        # ── Placeholder state ──────────────────────────────────────────────
        elif "last_result" not in st.session_state:
            st.markdown("""
            <div style="text-align:center; padding:4rem 2rem; color:#3D4258;">
              <p style="font-size:2rem; margin-bottom:0.5rem;">🧠</p>
              <p style="font-size:0.85rem;">Enter text above and click <strong>Analyse Text</strong></p>
              <p style="font-size:0.75rem; margin-top:0.3rem;">
                Or select an example from the sidebar →
              </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Show cached last result ────────────────────────────────────────
        elif "last_result" in st.session_state and not run_btn:
            _render_results(
                **st.session_state["last_result"],
                show_preprocessing=show_preprocessing,
                show_confidence=show_confidence,
            )

    # ══════════════════════════════════════════
    # TAB 2 — DATASET INSIGHTS
    # ══════════════════════════════════════════
    with tab_dataset:
        _render_dataset_tab()


def _run_analysis(
    user_input: str,
    predict_fn,
    analyze_sent_fn,
    show_preprocessing: bool,
    show_confidence: bool,
):
    """
    Validate input, run the full inference pipeline, and render results.
    Caches the result in session_state for persistence between reruns.

    Args:
        user_input (str):        Raw text from the text area.
        predict_fn (callable):   predict() from predictor.py.
        analyze_sent_fn (callable): VADER analyser closure.
        show_preprocessing (bool): Toggle from UI.
        show_confidence (bool):    Toggle from UI.
    """
    # ── Validate ──────────────────────────────────────────────────────────
    if not user_input or not user_input.strip():
        st.warning("Please enter some text before analysing.")
        return

    if len(user_input.strip().split()) < 5:
        st.warning("Text is very short — results may be unreliable. "
                   "Try adding more context.")

    # ── Run pipeline with progress indicator ──────────────────────────────
    with st.spinner("Running analysis..."):
        progress = st.progress(0, text="Preprocessing text...")
        time.sleep(0.05)

        # Prediction (includes preprocessing + vectorization internally)
        prediction = predict_fn(user_input)
        progress.progress(40, text="Running model...")
        time.sleep(0.05)

        # Sentiment (on raw text — VADER works best on unprocessed text)
        sentiment = analyze_sent_fn(user_input)
        progress.progress(70, text="Computing severity...")
        time.sleep(0.05)

        # For severity + preprocessing display, get clean_text
        from src.preprocessing.text_processor import preprocess
        clean_text = preprocess(user_input)
        severity   = run_severity(clean_text, sentiment["sentiment_score"])
        progress.progress(100, text="Done.")
        time.sleep(0.15)
        progress.empty()

    # ── Handle empty-after-preprocessing case ─────────────────────────────
    if prediction["predicted_class"] is None:
        st.error(
            "The text became empty after preprocessing. "
            "Please try a longer or more descriptive post."
        )
        return

    # ── Cache result in session state ─────────────────────────────────────
    result = dict(
        prediction=prediction,
        sentiment=sentiment,
        severity=severity,
        raw_text=user_input,
        clean_text=clean_text,
    )
    st.session_state["last_result"] = result

    _render_results(
        **result,
        show_preprocessing=show_preprocessing,
        show_confidence=show_confidence,
    )


def _render_results(
    prediction: dict,
    sentiment: dict,
    severity: float,
    raw_text: str,
    clean_text: str,
    show_preprocessing: bool = False,
    show_confidence: bool = True,
):
    """
    Render the full results panel for a completed analysis.

    Args:
        prediction (dict):  Output of predictor.predict().
        sentiment (dict):   {sentiment_score, sentiment_label}.
        severity (float):   Severity score in [0.0, 1.0].
        raw_text (str):     Original user input.
        clean_text (str):   Preprocessed text.
        show_preprocessing (bool): Show preprocessing expander.
        show_confidence (bool):    Show confidence chart.
    """
    predicted = prediction["predicted_class"]
    st.markdown("---")

    # ── Row 1: Condition + Confidence chart ───────────────────────────────
    if show_confidence and prediction["all_scores"]:
        col_cond, col_chart = st.columns([1, 1])
        with col_cond:
            render_condition_card(predicted, prediction["confidence"])
        with col_chart:
            st.markdown(
                '<div class="ms-card"><p class="ms-section-label">Confidence / Decision Scores</p>',
                unsafe_allow_html=True,
            )
            fig = build_confidence_chart(prediction["all_scores"], predicted)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        render_condition_card(predicted, prediction["confidence"])

    # ── Row 2: Sentiment + Severity ───────────────────────────────────────
    col_sent, col_sev = st.columns(2)
    with col_sent:
        render_sentiment_card(sentiment["sentiment_score"], sentiment["sentiment_label"])
    with col_sev:
        render_severity_card(severity)

    # ── Row 3: Summary metrics ────────────────────────────────────────────
    word_count = len(clean_text.split())
    meta       = CLASS_META.get(predicted, {})
    col_a, col_b, col_c, col_d = st.columns(4)
    metrics = [
        (col_a, "Condition", f"{meta.get('emoji','●')} {meta.get('name', predicted.title())}", meta.get("color","#7A6BAE")),
        (col_b, "Sentiment Score", f"{sentiment['sentiment_score']:+.4f}", SENTIMENT_COLORS.get(sentiment['sentiment_label'], '#5B8DB8')),
        (col_c, "Severity", f"{severity:.4f}", severity_color(severity)),
        (col_d, "Token Count", str(word_count), "#5B6280"),
    ]
    for col, label, val, color in metrics:
        with col:
            st.markdown(f"""
            <div class="ms-metric">
              <div class="ms-metric-val" style="color:{color};">{val}</div>
              <div class="ms-metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Preprocessing detail expander ─────────────────────────────────────
    if show_preprocessing:
        render_processing_detail(raw_text, clean_text, word_count)


def _render_dataset_tab():
    """
    Render the Dataset Insights tab showing pre-computed training data stats.

    Loads sentiment_stats.json, severity_stats.json, and model_comparison.csv
    saved by the pipeline and presents them as interactive charts and tables.
    Shows a friendly message if stats haven't been generated yet.
    """
    sentiment_stats = load_eda_stats()
    severity_stats  = load_severity_stats()
    comparison_df   = load_model_comparison()

    if not sentiment_stats and not severity_stats and comparison_df is None:
        st.info(
            "📊 Dataset insights haven't been generated yet.\n\n"
            "Run the full pipeline to create them:\n\n"
            "```bash\npython run_pipeline.py\npython run_pipeline.py --advanced-only\n```"
        )
        return

    st.markdown(
        '<p class="ms-section-label" style="margin-top:1rem;">Training Dataset Analysis</p>',
        unsafe_allow_html=True,
    )

    # ── Overview chart ─────────────────────────────────────────────────────
    if sentiment_stats and severity_stats:
        fig = build_dataset_overview_chart(sentiment_stats, severity_stats)
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

        # Summary cards row
        col1, col2, col3 = st.columns(3)
        mn = sentiment_stats.get("most_negative_class", "N/A")
        ms = severity_stats.get("most_severe_class", "N/A")
        sent_by_class = sentiment_stats.get("by_class", {})

        with col1:
            mn_color = config.CLASS_COLORS.get(mn, "#AAAAAA")
            st.markdown(f"""
            <div class="ms-card ms-card-accent" style="--accent:{mn_color};">
              <p class="ms-section-label">Most Negative Sentiment</p>
              <p style="font-size:1.4rem; color:{mn_color}; margin:0.2rem 0;">
                {CLASS_META.get(mn,{}).get('emoji','●')} {CLASS_META.get(mn,{}).get('name', mn)}
              </p>
              <p style="color:#5B6280; font-size:0.75rem; margin:0;">
                Mean score: {sent_by_class.get(mn,{}).get('mean_score',0):+.4f}
              </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            ms_color = config.CLASS_COLORS.get(ms, "#AAAAAA")
            sev_by   = severity_stats.get("by_class", {})
            st.markdown(f"""
            <div class="ms-card ms-card-accent" style="--accent:{ms_color};">
              <p class="ms-section-label">Highest Severity</p>
              <p style="font-size:1.4rem; color:{ms_color}; margin:0.2rem 0;">
                {CLASS_META.get(ms,{}).get('emoji','●')} {CLASS_META.get(ms,{}).get('name', ms)}
              </p>
              <p style="color:#5B6280; font-size:0.75rem; margin:0;">
                Mean severity: {sev_by.get(ms,{}).get('mean_severity',0):.4f}
              </p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            total_posts = sum(
                v.get("n_posts", 0)
                for v in sent_by_class.values()
            )
            st.markdown(f"""
            <div class="ms-card">
              <p class="ms-section-label">Dataset Size</p>
              <p style="font-size:1.4rem; color:#C8CCDA; margin:0.2rem 0;">
                {total_posts:,}
              </p>
              <p style="color:#5B6280; font-size:0.75rem; margin:0;">
                Posts across {len(config.CLASSES)} classes
              </p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Per-class sentiment table
        st.markdown('<p class="ms-section-label">Per-Class Sentiment Statistics</p>',
                    unsafe_allow_html=True)
        rows = []
        for cls in config.CLASSES:
            s = sent_by_class.get(cls, {})
            sv = severity_stats.get("by_class", {}).get(cls, {})
            rows.append({
                "Class":           CLASS_META[cls]["name"],
                "Mean Sentiment":  f"{s.get('mean_score',0):+.4f}",
                "% Negative":      f"{s.get('pct_negative',0):.1f}%",
                "% Positive":      f"{s.get('pct_positive',0):.1f}%",
                "Mean Severity":   f"{sv.get('mean_severity',0):.4f}",
                "High Severity %": f"{sv.get('high_severity_pct',0):.1f}%",
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    # ── Model comparison ────────────────────────────────────────────────────
    if comparison_df is not None:
        st.divider()
        st.markdown('<p class="ms-section-label">Model Performance Comparison</p>',
                    unsafe_allow_html=True)

        best_model = comparison_df.iloc[0]["model"]
        best_f1    = comparison_df.iloc[0]["f1_macro"]
        best_color = "#F0C040"

        st.markdown(
            f'<p style="color:{best_color}; font-size:0.85rem; margin-bottom:0.6rem;">'
            f'🏆 Best model: <strong>{best_model.replace("_"," ").title()}</strong>'
            f' — Macro F1: {best_f1:.4f}</p>',
            unsafe_allow_html=True,
        )

        metrics = ["accuracy", "precision", "recall", "f1_macro"]
        fig = go.Figure()
        metric_colors = ["#5B8DB8", "#4FAF8C", "#E07B54", "#7A6BAE"]
        for col, color in zip(metrics, metric_colors):
            fig.add_trace(go.Bar(
                name=col.replace("_", " ").title(),
                x=comparison_df["model"].str.replace("_", " ").str.title(),
                y=comparison_df[col],
                marker_color=color,
                text=comparison_df[col].apply(lambda v: f"{v:.3f}"),
                textposition="outside",
                textfont=dict(size=10),
            ))

        fig.update_layout(
            barmode="group",
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Mono, monospace", color="#C8CCDA", size=10),
            legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor="#1E2235", range=[0, 1.1]),
            xaxis=dict(showgrid=False),
            bargap=0.15,
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

        # Raw table
        display_df = comparison_df.copy()
        display_df["model"] = display_df["model"].str.replace("_", " ").str.title()
        display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()

"""
app.py  (project root)
───────────────────────
Thin shim that re-exports the FastAPI app from api/app.py.
 
This lets you run:
    uvicorn app:app --reload
 
from the project root directory, which ensures that all
`src.*` imports resolve correctly (Python path includes cwd).
 
The actual application logic lives in api/app.py.
"""
 
from api.app import app  # noqa: F401 — re-export for uvicorn