"""
predictor.py — Inference module for the MindScope pipeline.

Responsibility (ONLY):
    Load the best saved model + vectorizer + label encoder, accept raw
    text input, run it through the preprocessing and feature pipeline,
    and return a structured prediction result.

This module never trains or evaluates. It is the production-facing
inference endpoint — wrappable directly in a FastAPI handler.

Input:  raw_text: str  (any Reddit post, unprocessed)
Output: dict with predicted class, confidence, and all class scores

Usage:
    >>> from src.modeling.predictor import predict
    >>> result = predict("I can't stop checking the locks, my mind won't rest")
    >>> result["predicted_class"]
    'ocd'
"""

from src.utils import config
from src.utils.helpers import load_object
from src.utils.logger import get_logger
from src.preprocessing.text_processor import preprocess

logger = get_logger(__name__)

# ── Module-level cache: load artifacts once, reuse across calls ────────────
_model      = None
_vectorizer = None
_label_enc  = None


def _load_artifacts():
    """
    Load model, vectorizer, and label encoder from disk into module cache.

    Called lazily on the first predict() call. Subsequent calls reuse the
    in-memory objects without touching disk.

    Raises:
        FileNotFoundError: Via helpers.load_object if any artifact is missing.
    """
    global _model, _vectorizer, _label_enc

    if _model is not None:
        return   # already loaded

    logger.info("Loading inference artifacts from disk...")
    _model      = load_object(config.BEST_MODEL_PATH)
    _vectorizer = load_object(config.FEATURE_VEC_PATH)
    _label_enc  = load_object(config.LABEL_ENCODER_PATH)

    with open(config.BEST_MODEL_NAME_PATH) as f:
        best_name = f.read().strip()
    logger.info(f"Loaded model: {best_name}")


def predict(raw_text: str) -> dict:
    """
    Run end-to-end inference on a raw text string.

    Pipeline:
        1. Preprocess raw_text (text_processor.preprocess).
        2. Vectorize with the saved TF-IDF vectorizer.
        3. Predict with the best saved model.
        4. Return class label + confidence + all class scores.

    For LinearSVC (no predict_proba): decision_function scores are
    returned as-is (not probabilities). For models with predict_proba,
    true probabilities are returned.

    Args:
        raw_text (str): Unprocessed post text (title + body or any string).

    Returns:
        dict: {
            "predicted_class": str,          # e.g. "depression"
            "confidence": float | None,      # probability if available
            "all_scores": dict[str, float]   # one score per class
        }

    Example:
        >>> predict("I've been feeling hopeless and can't get out of bed")
        {'predicted_class': 'depression', 'confidence': 0.91, 'all_scores': {...}}
    """
    _load_artifacts()

    clean = preprocess(raw_text)
    if not clean.strip():
        logger.warning("Input text became empty after preprocessing.")
        return {"predicted_class": None, "confidence": None, "all_scores": {}}

    X = _vectorizer.transform([clean])

    predicted_label = _model.predict(X)[0]

    # Confidence: use predict_proba if available, else decision_function
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X)[0]
        scores = dict(zip(config.CLASSES, [round(float(p), 4) for p in proba]))
        confidence = round(float(max(proba)), 4)
    else:
        # LinearSVC: decision_function returns raw margin scores
        decision = _model.decision_function(X)[0]
        classes  = list(_model.classes_) if hasattr(_model, "classes_") else config.CLASSES
        scores   = dict(zip(classes, [round(float(d), 4) for d in decision]))
        confidence = None   # not a probability

    return {
        "predicted_class": predicted_label,
        "confidence":      confidence,
        "all_scores":      scores,
    }


def predict_batch(texts: list) -> list:
    """
    Run inference on a list of raw text strings.

    More efficient than calling predict() in a loop because vectorization
    is done in one batch transform call.

    Args:
        texts (list[str]): List of raw post strings.

    Returns:
        list[dict]: One result dict per input text (same order).

    Example:
        >>> results = predict_batch(["feeling down", "can't focus"])
        >>> [r["predicted_class"] for r in results]
        ['depression', 'adhd']
    """
    _load_artifacts()

    from src.preprocessing.text_processor import preprocess_series
    import pandas as pd

    series = pd.Series(texts)
    cleaned = preprocess_series(series).tolist()

    X = _vectorizer.transform(cleaned)
    predictions = _model.predict(X)

    results = []
    if hasattr(_model, "predict_proba"):
        probas = _model.predict_proba(X)
        for pred, proba in zip(predictions, probas):
            scores     = dict(zip(config.CLASSES, [round(float(p), 4) for p in proba]))
            confidence = round(float(max(proba)), 4)
            results.append({"predicted_class": pred, "confidence": confidence,
                            "all_scores": scores})
    else:
        decisions = _model.decision_function(X)
        classes   = list(_model.classes_) if hasattr(_model, "classes_") else config.CLASSES
        for pred, decision in zip(predictions, decisions):
            scores = dict(zip(classes, [round(float(d), 4) for d in decision]))
            results.append({"predicted_class": pred, "confidence": None,
                            "all_scores": scores})

    return results