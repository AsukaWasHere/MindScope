"""
api/model_loader.py
────────────────────
Loads and caches all ML artifacts needed for inference.

Design decisions
────────────────
  - All loading happens ONCE at FastAPI startup via lifespan events.
    After that, the loaded objects live in a module-level registry dict
    so every request reuses the same in-memory objects — no re-loading.

  - Supports two backends, selected by API_CONFIG["model_backend"]:
      "classical" → TF-IDF vectorizer  +  a scikit-learn / LightGBM model
      "bert"      → HuggingFace tokenizer  +  BertForSequenceClassification

  - Follows separation-of-concerns: this module only LOADS artifacts,
    it does not run preprocessing or inference (those live in predictor.py).
"""

import os

from src.utils.config import PATHS, BERT_CONFIG, API_CONFIG, CLASSES
from src.utils.helpers import load_object, get_label_encoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Module-level registry — populated once at startup ─────────────────
_registry: dict = {}


# ══════════════════════════════════════════════════════════════════════
def load_artifacts() -> None:
    """
    Load all model artifacts into the in-memory registry.
    Called once during FastAPI lifespan startup.
    """
    backend = API_CONFIG["model_backend"]
    logger.info(f"[model_loader] Loading artifacts for backend: '{backend}'")

    # Label encoder is always needed
    _registry["label_encoder"] = get_label_encoder(CLASSES)

    if backend == "classical":
        _load_classical()
    elif backend == "bert":
        _load_bert()
    else:
        raise ValueError(
            f"Unknown model_backend '{backend}'. "
            "Choose 'classical' or 'bert' in API_CONFIG."
        )

    logger.info("[model_loader] ✓ All artifacts loaded and ready")


def get_registry() -> dict:
    """Return the loaded artifact registry. Raises if not yet initialised."""
    if not _registry:
        raise RuntimeError(
            "Artifacts not loaded. Ensure load_artifacts() ran at startup."
        )
    return _registry


# ══════════════════════════════════════════════════════════════════════
# Private loaders
# ══════════════════════════════════════════════════════════════════════

def _load_classical() -> None:
    """Load TF-IDF vectorizer + a trained sklearn/LightGBM model."""
    model_name = API_CONFIG["classical_model_name"]
    model_path = os.path.join(PATHS["models_dir"], f"{model_name}.pkl")
    vectorizer_path = PATHS["tfidf_vectorizer"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            f"Run the pipeline first:  python run_pipeline.py --model {model_name}"
        )
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found at: {vectorizer_path}\n"
            "Run the pipeline first."
        )

    logger.info(f"[model_loader] Loading classical model: {model_path}")
    _registry["model"]      = load_object(model_path)
    _registry["vectorizer"] = load_object(vectorizer_path)
    _registry["backend"]    = "classical"


def _load_bert() -> None:
    """Load fine-tuned BERT tokenizer + model weights."""

    # ✅ LAZY IMPORTS (only change)
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer

    save_dir = BERT_CONFIG["save_dir"]

    if not os.path.exists(save_dir):
        raise FileNotFoundError(
            f"BERT model directory not found at: {save_dir}\n"
            "Run the pipeline first:  python run_pipeline.py --use_bert"
        )

    logger.info(f"[model_loader] Loading BERT from: {save_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(save_dir)
    model     = BertForSequenceClassification.from_pretrained(save_dir)
    model.to(device)
    model.eval()

    _registry["tokenizer"] = tokenizer
    _registry["model"]     = model
    _registry["device"]    = device
    _registry["backend"]   = "bert"