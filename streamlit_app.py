"""
streamlit_app.py
─────────────────
Streamlit demo UI for MindScope.
Calls the /predict endpoint of the FastAPI backend.

Run locally:
    streamlit run streamlit_app.py

On HuggingFace Spaces or Render, this file is the entry point.
Set the FASTAPI_URL environment variable to point at your live API.
If not set, it falls back to localhost for local dev.
"""

import os
import requests
import streamlit as st
import plotly.graph_objects as go

# ── Config ─────────────────────────────────────────────────────────────
API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

CONDITION_DESCRIPTIONS = {
    "adhd":       "ADHD — Attention deficit, impulsivity, executive dysfunction",
    "aspergers":  "Asperger's — Social difficulty, sensory sensitivity, ASD",
    "depression": "Depression — Persistent low mood, hopelessness, anhedonia",
    "ocd":        "OCD — Intrusive thoughts, compulsions, rituals",
    "ptsd":       "PTSD — Trauma, flashbacks, hypervigilance",
}

SEVERITY_COLORS = {"low": "#2ecc71", "moderate": "#f39c12", "high": "#e74c3c"}
SENTIMENT_EMOJI = {"positive": "😊", "negative": "😔", "neutral": "😐"}

# ── Page setup ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindScope — Mental Health NLP",
    page_icon="🧠",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif; }

    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        font-size: 15px;
        line-height: 1.6;
    }
    .result-card {
        background: #f8f9fa;
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        border-left: 5px solid #6c5ce7;
    }
    .severity-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .disclaimer {
        background: #fff3cd;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 13px;
        color: #856404;
        border: 1px solid #ffc107;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════
st.markdown("# 🧠 MindScope")
st.markdown("#### Mental Health NLP Classification Demo")
st.markdown("---")

st.markdown("""
<div class="disclaimer">
⚠️ <strong>Research tool only.</strong> This is not a diagnostic instrument.
Predictions are based on Reddit community language patterns and must never
be used to label or make decisions about real individuals.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# Input
# ══════════════════════════════════════════════════════════════════════
st.markdown("### Enter a post to analyse")

example_texts = {
    "Select an example...": "",
    "Depression example":   "I haven't felt happy in months. Nothing interests me anymore and I can't seem to get out of bed.",
    "ADHD example":         "I literally cannot focus on anything for more than 2 minutes. My brain jumps around constantly.",
    "OCD example":          "I keep having intrusive thoughts and I feel like I need to check the locks 20 times before sleeping.",
    "PTSD example":         "I keep having flashbacks to what happened. Loud noises send me into a panic.",
    "Asperger's example":   "Social situations are exhausting. I can never read people's expressions correctly.",
}

selected = st.selectbox("Or load an example:", list(example_texts.keys()))
default_text = example_texts[selected]

user_text = st.text_area(
    "Post text:",
    value=default_text,
    height=150,
    placeholder="Describe what you're experiencing... (minimum 10 words)",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 4])
with col1:
    analyse_btn = st.button("🔍 Analyse", use_container_width=True, type="primary")
with col2:
    st.markdown("")   # spacer


# ══════════════════════════════════════════════════════════════════════
# Prediction
# ══════════════════════════════════════════════════════════════════════
if analyse_btn:
    if len(user_text.split()) < 5:
        st.warning("Please enter at least 5 words for a meaningful prediction.")
        st.stop()

    with st.spinner("Running inference..."):
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json={"text": user_text},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error(
                f"❌ Could not connect to the API at **{API_URL}**.\n\n"
                "Make sure the FastAPI server is running:\n"
                "```\nuvicorn app:app --reload\n```"
            )
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e}")
            st.stop()

    # ── Results layout ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Results")

    # Top row: prediction + severity + sentiment
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Predicted Condition**")
        st.markdown(f"### {data['prediction'].upper()}")
        st.caption(CONDITION_DESCRIPTIONS.get(data["prediction"], ""))

    with c2:
        sev   = data["severity"]
        color = SEVERITY_COLORS[sev]
        st.markdown("**Severity Estimate**")
        st.markdown(
            f'<span class="severity-badge" style="background:{color}20;color:{color}">'
            f'{sev.upper()}</span>',
            unsafe_allow_html=True,
        )
        st.caption("Heuristic — not clinical")

    with c3:
        sent = data["sentiment"]
        st.markdown("**Sentiment**")
        st.markdown(f"### {SENTIMENT_EMOJI[sent]} {sent.capitalize()}")
        compound = data["sentiment_scores"]["compound"]
        st.caption(f"VADER compound: {compound:+.3f}")

    # Confidence bar chart
    st.markdown("#### Confidence across all conditions")
    scores      = data["all_scores"]
    labels      = list(scores.keys())
    values      = list(scores.values())
    bar_colors  = [
        "#6c5ce7" if lbl == data["prediction"] else "#dfe6e9"
        for lbl in labels
    ]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=bar_colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"),
        xaxis_title="Condition",
        margin=dict(t=20, b=20),
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment detail
    with st.expander("📊 Full sentiment breakdown (VADER)"):
        ss = data["sentiment_scores"]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Negative", f"{ss['neg']:.3f}")
        s2.metric("Neutral",  f"{ss['neu']:.3f}")
        s3.metric("Positive", f"{ss['pos']:.3f}")
        s4.metric("Compound", f"{ss['compound']:+.3f}")

    # Preprocessed text
    with st.expander("🔧 Preprocessed text (what the model sees)"):
        st.code(data["clean_text"], language=None)

    # Raw JSON
    with st.expander("📋 Raw API response (JSON)"):
        st.json(data)


# ══════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "MindScope — NLP pipeline built with scikit-learn, HuggingFace, FastAPI & Streamlit. "
    "Data sourced from Reddit mental health communities."
)