"""
trainer.py — Model training for the MindScope pipeline.

Responsibility (ONLY):
    Accept pre-built feature matrices (X_train, y_train), instantiate
    each model, train it, and save the fitted object to disk.
    No evaluation happens here — that is evaluator.py's job.

Models trained:
    A. Logistic Regression  (sklearn)
    B. Multinomial Naive Bayes (sklearn)
    C. Linear SVM / LinearSVC  (sklearn)

All hyperparameters live in config.py. The module exposes a clean
public API of four functions so run_pipeline.py can orchestrate it
without knowing sklearn internals.

Input:  X_train (sparse), y_train (pd.Series or array)
Output: models/saved/{model_name}.pkl   (one per model)
        models/saved/label_encoder.pkl
        models/saved/best_model.pkl
        models/saved/best_model_name.txt
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from src.utils import config
from src.utils.helpers import ensure_dir, get_label_encoder, save_object
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def train_model(name: str, X_train, y_train, tune: bool = False):
    """
    Instantiate and train a single model by name.

    Supported names (must match config.MODEL_NAMES):
        "logistic_regression"
        "naive_bayes"
        "svm"

    Args:
        name (str):    Model key from config.MODEL_NAMES.
        X_train:       Sparse feature matrix (output of tfidf_vectorizer).
        y_train:       Array-like of string class labels.
        tune (bool):   If True, run GridSearchCV before final training.
                       Slower but finds better hyperparameters.

    Returns:
        Fitted sklearn estimator (best estimator if tune=True).

    Raises:
        ValueError: If name is not in config.MODEL_NAMES.

    Example:
        >>> model = train_model("logistic_regression", X_train, y_train)
    """
    if name not in config.MODEL_NAMES:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {config.MODEL_NAMES}"
        )

    logger.info(f"── Training: {name} {'(+ GridSearch)' if tune else ''}")
    t0 = time.time()

    model = _build_model(name)

    if tune:
        model = _tune_model(name, model, X_train, y_train)
    else:
        model.fit(X_train, y_train)

    elapsed = round(time.time() - t0, 2)
    logger.info(f"   Training complete in {elapsed}s")

    return model


def train_all_models(X_train, y_train, tune: bool = False) -> dict:
    """
    Train every model in config.MODEL_NAMES and save each to disk.

    Also fits and saves the LabelEncoder used consistently across
    training, evaluation, and inference.

    Args:
        X_train:     Sparse feature matrix.
        y_train:     Array-like of string class labels.
        tune (bool): Run GridSearchCV for each model if True.

    Returns:
        dict: {model_name: fitted_model} for all trained models.

    Example:
        >>> models = train_all_models(X_train, y_train)
        >>> models.keys()
        dict_keys(['logistic_regression', 'naive_bayes', 'svm'])
    """
    logger.info("=== Stage 6: Model Training ===")
    ensure_dir(config.MODELS_DIR)

    # Save the label encoder once — used by all downstream stages
    le = get_label_encoder(config.CLASSES)
    save_object(le, config.LABEL_ENCODER_PATH)
    logger.info(f"Label encoder saved: {config.LABEL_ENCODER_PATH}")

    trained_models = {}
    for name in config.MODEL_NAMES:
        model = train_model(name, X_train, y_train, tune=tune)
        save_model(name, model)
        trained_models[name] = model

    logger.info(f"All {len(trained_models)} models trained and saved.")
    return trained_models


def save_model(name: str, model) -> None:
    """
    Persist a single fitted model to models/saved/<name>.pkl.

    Uses the path defined in config.MODEL_PATHS[name] so there
    is no path logic anywhere outside config.py.

    Args:
        name (str): Model key (e.g., "logistic_regression").
        model:      Fitted sklearn estimator.

    Example:
        >>> save_model("svm", fitted_svm)
    """
    path = config.MODEL_PATHS[name]
    save_object(model, path)
    logger.info(f"Model saved: {path}")


def save_best_model(name: str, model) -> None:
    """
    Write the best model to models/saved/best_model.pkl and record
    its name in models/saved/best_model_name.txt.

    Called by compare_models() in evaluator.py after all models are
    evaluated. Predictor.py loads from these fixed paths.

    Args:
        name (str): Name of the best model (e.g., "svm").
        model:      Fitted sklearn estimator for the best model.

    Example:
        >>> save_best_model("svm", trained_svm)
    """
    save_object(model, config.BEST_MODEL_PATH)
    with open(config.BEST_MODEL_NAME_PATH, "w") as f:
        f.write(name)
    logger.info(f"Best model '{name}' saved to: {config.BEST_MODEL_PATH}")


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _build_model(name: str):
    """
    Construct an unfitted sklearn estimator for the given model name.

    All constructor arguments come from config.py — nothing is hardcoded here.

    Models and their rationale:
        logistic_regression — Strong linear baseline for text; fast, interpretable.
                              C=5.0 gives mild regularisation, good for high-dim TF-IDF.
        naive_bayes         — MultinomialNB is the classic text classifier.
                              Requires non-negative features (TF-IDF ≥ 0 ✓).
                              alpha=0.1 (light Laplace smoothing) outperforms alpha=1.
        svm                 — LinearSVC consistently beats LR on high-dim sparse text.
                              Maximises the margin — robust to class overlap.

    Args:
        name (str): One of config.MODEL_NAMES.

    Returns:
        Unfitted sklearn estimator.
    """
    if name == "logistic_regression":
        return LogisticRegression(
            C=config.LR_C,
            max_iter=config.LR_MAX_ITER,
            solver=config.LR_SOLVER,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )

    if name == "naive_bayes":
        return MultinomialNB(alpha=config.NB_ALPHA)

    if name == "svm":
        return LinearSVC(
            C=config.SVM_C,
            max_iter=config.SVM_MAX_ITER,
            random_state=config.RANDOM_SEED,
        )

    # Unreachable if caller uses train_model() which validates name first
    raise ValueError(f"No builder registered for model '{name}'.")


def _tune_model(name: str, model, X_train, y_train):
    """
    Run GridSearchCV over the parameter grid defined in config.GRIDSEARCH_PARAMS.

    Searches using macro-F1 scoring (handles class imbalance correctly).
    Returns the best estimator refitted on the full X_train.

    Args:
        name (str):  Model key for looking up the param grid.
        model:       Unfitted sklearn estimator.
        X_train:     Sparse feature matrix.
        y_train:     Array-like class labels.

    Returns:
        Best fitted estimator from GridSearchCV.
    """
    param_grid = config.GRIDSEARCH_PARAMS.get(name, {})
    if not param_grid:
        logger.warning(f"No param grid defined for '{name}'. Skipping GridSearch.")
        model.fit(X_train, y_train)
        return model

    logger.info(
        f"   GridSearchCV: cv={config.GRIDSEARCH_CV}, "
        f"scoring={config.GRIDSEARCH_SCORING}, grid={param_grid}"
    )
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=config.GRIDSEARCH_CV,
        scoring=config.GRIDSEARCH_SCORING,
        n_jobs=-1,
        verbose=0,
        refit=True,   # refit best params on full X_train automatically
    )
    gs.fit(X_train, y_train)
    logger.info(f"   Best params : {gs.best_params_}")
    logger.info(f"   Best CV F1  : {gs.best_score_:.4f}")
    return gs.best_estimator_