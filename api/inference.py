"""
api/inference.py
─────────────────
Runs the full inference pipeline for a single raw text input.

Responsibilities
────────────────
  1. Preprocess raw text using the existing text_processor.preprocess()
  2. Vectorize / tokenize depending on the active backend
  3. Run model forward pass → predicted class + confidence scores
  4. Run VADER sentiment analysis → polarity + compound score
  5. Derive a severity rating from sentiment + prediction confidence
  6. Return a structured dict consumed by the /predict endpoint

This module does NOT load models (model_loader.py does that).
This module does NOT define API routes (app.py does that).
"""

try:
    import torch
except ImportError:
    torch = None
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.preprocessing.text_processor import preprocess
from src.utils.config import BERT_CONFIG
from src.utils.logger import get_logger
from api.model_loader import get_registry

logger = get_logger(__name__)

# VADER analyser is stateless — instantiate once at module level
_vader = SentimentIntensityAnalyzer()


# ══════════════════════════════════════════════════════════════════════
def run_inference(raw_text: str) -> dict:
    """
    End-to-end inference for a single piece of raw text.

    Parameters
    ----------
    raw_text : str — uncleaned post text from the API request

    Returns
    -------
    dict with keys:
        prediction   : str   — predicted condition label
        confidence   : float — probability of predicted class (0–1)
        all_scores   : dict  — probability for each of the 5 classes
        sentiment    : str   — "positive" | "negative" | "neutral"
        sentiment_scores : dict — VADER neg/neu/pos/compound breakdown
        severity     : str   — "low" | "moderate" | "high"
        clean_text   : str   — preprocessed text (useful for debugging)
    """
    registry      = get_registry()
    label_encoder = registry["label_encoder"]
    backend       = registry["backend"]

    # ── Step 1: Preprocess ────────────────────────────────────────────
    clean_text = preprocess(raw_text)
    logger.info(f"[inference] clean_text (first 80 chars): {clean_text[:80]!r}")

    # ── Step 2: Predict ───────────────────────────────────────────────
    if backend == "classical":
        prediction, confidence, all_scores = _predict_classical(
            clean_text, registry, label_encoder
        )
    else:
        prediction, confidence, all_scores = _predict_bert(
            raw_text, registry, label_encoder      # BERT gets raw text; it tokenizes internally
        )

    # ── Step 3: Sentiment analysis (VADER) ───────────────────────────
    sentiment_label, sentiment_scores = _analyse_sentiment(raw_text)

    # ── Step 4: Severity score ────────────────────────────────────────
    severity = _compute_severity(sentiment_scores["compound"], confidence, prediction)

    logger.info(
        f"[inference] → prediction={prediction}  confidence={confidence:.3f}  "
        f"sentiment={sentiment_label}  severity={severity}"
    )

    return {
        "prediction":       prediction,
        "confidence":       round(confidence, 4),
        "all_scores":       {k: round(v, 4) for k, v in all_scores.items()},
        "sentiment":        sentiment_label,
        "sentiment_scores": {k: round(v, 4) for k, v in sentiment_scores.items()},
        "severity":         severity,
        "clean_text":       clean_text,
    }


# ══════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════

def _predict_classical(clean_text: str, registry: dict, label_encoder) -> tuple:
    """TF-IDF vectorization → sklearn/LightGBM prediction."""
    vectorizer = registry["vectorizer"]
    model      = registry["model"]

    X = vectorizer.transform([clean_text])          # sparse (1, vocab)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    else:
        # LinearSVC: use decision_function + softmax normalisation
        scores = model.decision_function(X)[0]
        exp_scores = np.exp(scores - np.max(scores))
        proba = exp_scores / exp_scores.sum()

    class_idx   = int(np.argmax(proba))
    confidence  = float(proba[class_idx])
    prediction  = label_encoder.inverse_transform([class_idx])[0]
    all_scores  = dict(zip(label_encoder.classes_, proba.tolist()))

    return prediction, confidence, all_scores


def _predict_bert(raw_text: str, registry: dict, label_encoder) -> tuple:
    import torch
    """BERT tokenization → HuggingFace forward pass → softmax probabilities."""
    tokenizer = registry["tokenizer"]
    model     = registry["model"]
    device    = registry["device"]

    encoding = tokenizer(
        raw_text,
        max_length=BERT_CONFIG["max_length"],
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
        )

    proba      = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    class_idx  = int(np.argmax(proba))
    confidence = float(proba[class_idx])
    prediction = label_encoder.inverse_transform([class_idx])[0]
    all_scores = dict(zip(label_encoder.classes_, proba.tolist()))

    return prediction, confidence, all_scores


def _analyse_sentiment(text: str) -> tuple[str, dict]:
    """
    Run VADER sentiment analysis on raw (unprocessed) text.
    VADER works best on natural language, not lemmatized tokens.

    Returns
    -------
    label  : "positive" | "negative" | "neutral"
    scores : dict with neg, neu, pos, compound keys
    """
    scores   = _vader.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return label, scores


def _compute_severity(compound: float, confidence: float, prediction: str) -> str:
    """
    Derive a qualitative severity estimate from:
      - VADER compound score  (negative = more distress)
      - Model confidence      (higher confidence = clearer signal)
      - Predicted class       (some conditions carry higher baseline severity)

    Severity scale:  "low" | "moderate" | "high"

    Note: This is a heuristic for demo purposes only.
    It must NOT be used for clinical decisions.
    """
    # Classes associated with higher distress on average
    HIGH_SEVERITY_CLASSES = {"depression", "ptsd"}

    # Base score: negativity of sentiment mapped to 0–1
    negativity = max(0.0, -compound)        # compound in [-1, 1]; flip sign

    # Weight up if the predicted class is inherently high-severity
    class_weight = 0.2 if prediction in HIGH_SEVERITY_CLASSES else 0.0

    # Weight up by model confidence (a confident prediction is a clearer signal)
    score = negativity * confidence + class_weight

    if score >= 0.55:
        return "high"
    elif score >= 0.25:
        return "moderate"
    else:
        return "low"