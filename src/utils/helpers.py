"""
helpers.py — Shared utility functions for the MindScope pipeline.

Contains only generic, reusable helpers that multiple modules need.
No business logic lives here — only infrastructure utilities.

Modules that use this file:
    loader.py, cleaner.py, trainer.py, evaluator.py, predictor.py
"""

import os
import pickle

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# DIRECTORY UTILITIES
# ─────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """
    Create a directory (and any missing parents) if it does not already exist.

    This is called before saving any file to guarantee the target folder
    exists. Safe to call even if the directory already exists.

    Args:
        path (str): Absolute or relative path to the directory.

    Example:
        >>> ensure_dir("data/processed")
        # Creates data/processed/ if missing; does nothing if it exists.
    """
    os.makedirs(path, exist_ok=True)
    logger.debug(f"Directory ensured: {path}")


# ─────────────────────────────────────────────
# SERIALIZATION UTILITIES
# ─────────────────────────────────────────────

def save_object(obj: object, path: str) -> None:
    """
    Serialize a Python object to disk using pickle.

    Used to save trained models, fitted vectorizers, and label encoders
    so they can be loaded later without retraining.

    Args:
        obj (object): Any picklable Python object (model, vectorizer, etc.).
        path (str): Full file path where the object will be saved (e.g., .pkl).

    Example:
        >>> save_object(vectorizer, "models/saved/tfidf_vectorizer.pkl")
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Object saved to: {path}")


def load_object(path: str) -> object:
    """
    Deserialize a Python object from a pickle file on disk.

    Used to reload saved models and vectorizers for inference or evaluation
    without rerunning training.

    Args:
        path (str): Full file path to the saved pickle file.

    Returns:
        object: The deserialized Python object.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.

    Example:
        >>> vectorizer = load_object("models/saved/tfidf_vectorizer.pkl")
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved object found at: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Object loaded from: {path}")
    return obj


# ─────────────────────────────────────────────
# LABEL ENCODING UTILITIES
# ─────────────────────────────────────────────

def get_label_encoder(classes: list):
    """
    Return a fitted scikit-learn LabelEncoder for the given class list.

    Using a shared encoder ensures that label-to-integer mapping is
    consistent across training, evaluation, and inference stages.

    Args:
        classes (list): Ordered list of class label strings.
                        Should always come from config.CLASSES.

    Returns:
        sklearn.preprocessing.LabelEncoder: Fitted encoder.

    Example:
        >>> from src.utils import config
        >>> le = get_label_encoder(config.CLASSES)
        >>> le.transform(["adhd", "depression"])
        array([0, 2])
    """
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(classes)
    logger.debug(f"Label encoder fitted with classes: {classes}")
    return le