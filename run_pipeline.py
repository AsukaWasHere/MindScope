"""
run_pipeline.py — Single entrypoint to execute the full MindScope pipeline.

Usage:
    python run_pipeline.py                        # full run
    python run_pipeline.py --skip-eda             # skip EDA
    python run_pipeline.py --tune                 # enable GridSearchCV tuning
    python run_pipeline.py --model-only           # skip to training (uses cached features)

Stages:
    1  Data Ingestion       loader.py          → data/processed/merged_raw.csv
    2  Data Cleaning        cleaner.py         → data/processed/cleaned.csv
                                               → data/splits/{train,val,test}.csv
    3  Text Preprocessing   text_processor     → data/processed/processed_data.csv
    4  EDA                  eda.py             → reports/figures/*.png
                                               → reports/metrics/eda_stats.json
    5  Feature Engineering  tfidf_vectorizer   → data/features/{X,y}_{train,test}.*
                                               → models/saved/feature_tfidf_vectorizer.pkl
    6  Training + Eval      trainer/evaluator  → models/saved/*.pkl
                                               → reports/metrics/*.json | *.csv | *.txt
                                               → reports/figures/confusion_matrix_*.png
                                               → reports/figures/model_comparison_chart.png

Cache behaviour:
    Each stage checks that its output file exists AND contains data
    before deciding to skip. An empty or corrupted output file is treated
    the same as a missing one — the stage re-runs and overwrites it.
    Delete the relevant output file(s) to manually force a stage to re-run.

Bug fix (v2):
    Previous version checked only os.path.exists() for cache hits.
    This caused a crash when a CSV was created empty by a prior failed run.
    All cache checks now use _is_valid_csv() / _is_valid_pkl() which verify
    that the file exists AND has non-zero byte content.
"""

import argparse
import os
import sys

import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.ingestion.loader import load_data
from src.preprocessing.cleaner import clean_data, save_cleaned_data
from src.preprocessing.text_processor import preprocess_series
from src.analysis.eda import perform_eda
from src.features.tfidf_vectorizer import (
    create_tfidf_features, save_features, load_features,
)
from src.modeling.trainer import train_all_models
from src.modeling.evaluator import (
    evaluate_model, compare_models, save_evaluation_outputs,
)
from src.utils import config
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# CACHE VALIDATION HELPERS
# ─────────────────────────────────────────────

def _is_valid_csv(path: str, min_rows: int = 1) -> bool:
    """
    Return True only if `path` exists, is non-empty, and is a readable CSV
    with at least `min_rows` data rows (excluding the header).

    This prevents the pipeline from treating an empty file created by a
    previously failed run as a valid cache hit.

    Args:
        path (str):     Absolute path to the CSV file.
        min_rows (int): Minimum number of data rows required. Default 1.

    Returns:
        bool: True → file is valid and safe to load. False → must regenerate.
    """
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) == 0:
        logger.warning(f"[CACHE INVALID] File is empty: {path}")
        return False
    try:
        # Read only the first chunk — fast and avoids loading huge files
        sample = pd.read_csv(path, nrows=min_rows + 1)
        if len(sample) < min_rows:
            logger.warning(
                f"[CACHE INVALID] File has fewer than {min_rows} rows: {path}"
            )
            return False
        return True
    except Exception as exc:
        logger.warning(f"[CACHE INVALID] Cannot read CSV ({exc}): {path}")
        return False


def _is_valid_pkl(path: str) -> bool:
    """
    Return True only if `path` exists and has non-zero byte size.

    A zero-byte .pkl was written by a run that crashed before pickle.dump
    completed. We treat it as invalid so the stage that creates it re-runs.

    Args:
        path (str): Absolute path to the pickle file.

    Returns:
        bool: True → file is present and non-empty. False → must regenerate.
    """
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) == 0:
        logger.warning(f"[CACHE INVALID] Pickle file is empty: {path}")
        return False
    return True


def _is_valid_npz(path: str) -> bool:
    """
    Return True only if `path` is a non-empty .npz file.

    Args:
        path (str): Absolute path to the .npz sparse matrix file.

    Returns:
        bool: True → safe to load. False → must regenerate.
    """
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) == 0:
        logger.warning(f"[CACHE INVALID] NPZ file is empty: {path}")
        return False
    return True


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run(skip_eda: bool = False, tune: bool = False, model_only: bool = False, use_bert: bool = False, skip_classical: bool = False,) -> None:
    """
    Execute the full pipeline, respecting cache and CLI flags.

    Args:
        skip_eda (bool):   Skip Stage 4 EDA.
        tune (bool):       Enable GridSearchCV in Stage 6 training.
        model_only (bool): Jump straight to Stage 6 (assumes features exist).
    """
    logger.info("=" * 60)
    logger.info("MindScope Pipeline — Starting")
    logger.info("=" * 60)

    if not model_only:
        merged    = _stage_ingestion()
        cleaned   = _stage_cleaning(merged)
        processed = _stage_preprocessing(cleaned)
        if not skip_eda:
            _stage_eda(processed)
        else:
            logger.info("[SKIP] EDA skipped via --skip-eda.")
        X_train, X_test, y_train, y_test, _ = _stage_features(processed)
    else:
        logger.info("[SKIP] Stages 1–5 skipped via --model-only. Loading features...")
        X_train, X_test, y_train, y_test, _ = load_features()

    # ── Classical ML ─────────────────────────────
    if not skip_classical:
        _stage_train_eval(X_train, X_test, y_train, y_test, tune=tune)
    else:
        logger.info("[SKIP] Classical ML skipped via --skip_classical.")

    # ── BERT ─────────────────────────────────────
    if use_bert:
        _run_bert()

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info(f"  Best model     : {config.BEST_MODEL_PATH}")
    logger.info(f"  Comparison CSV : {config.COMPARISON_TABLE_PATH}")
    logger.info(f"  Metrics JSON   : {config.ALL_METRICS_PATH}")
    logger.info(f"  EDA figures    : {config.FIGURES_DIR}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────
# STAGE FUNCTIONS
# ─────────────────────────────────────────────

def _stage_ingestion() -> pd.DataFrame:
    """
    Stage 1 — Load raw CSVs or return cached merged_raw.csv.

    Cache is valid only when merged_raw.csv exists AND has ≥ 1 data row.
    An empty file (from a previous crashed run) causes a re-run.
    """
    if _is_valid_csv(config.MERGED_RAW_PATH, min_rows=1):
        logger.info("[CACHE] merged_raw.csv is valid — skipping ingestion.")
        return pd.read_csv(config.MERGED_RAW_PATH)

    logger.info("[STAGE 1] Running data ingestion...")
    return load_data()


def _stage_cleaning(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2 — Clean data and create splits, or load cached cleaned.csv.

    All four output files (cleaned.csv + 3 splits) must be valid CSVs.
    If any one is missing or empty the entire cleaning stage re-runs.
    """
    all_valid = (
        _is_valid_csv(config.CLEANED_DATA_PATH, min_rows=1)
        and _is_valid_csv(config.TRAIN_PATH, min_rows=1)
        and _is_valid_csv(config.VAL_PATH,   min_rows=1)
        and _is_valid_csv(config.TEST_PATH,   min_rows=1)
    )
    if all_valid:
        logger.info("[CACHE] cleaned.csv + splits are valid — skipping cleaning.")
        return pd.read_csv(config.CLEANED_DATA_PATH)

    logger.info("[STAGE 2] Running data cleaning + splitting...")
    cleaned = clean_data(merged)
    save_cleaned_data(cleaned)
    return cleaned


def _stage_preprocessing(cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 3 — Apply NLP preprocessing or load cached processed_data.csv.

    Cache is valid only when processed_data.csv exists, is non-empty,
    AND contains the `clean_text` column (proving it completed fully).
    """
    if _is_valid_csv(config.PROCESSED_DATA_PATH, min_rows=1):
        # Extra guard: verify the clean_text column actually exists
        header = pd.read_csv(config.PROCESSED_DATA_PATH, nrows=0)
        if config.CLEAN_TEXT_COLUMN in header.columns:
            logger.info(
                "[CACHE] processed_data.csv is valid — skipping preprocessing."
            )
            return pd.read_csv(config.PROCESSED_DATA_PATH)
        else:
            logger.warning(
                "[CACHE INVALID] processed_data.csv is missing the "
                f"`{config.CLEAN_TEXT_COLUMN}` column — re-running preprocessing."
            )

    logger.info("[STAGE 3] Running text preprocessing...")
    df = cleaned.copy()
    df[config.CLEAN_TEXT_COLUMN] = preprocess_series(df[config.TEXT_COLUMN])

    before = len(df)
    df = df[df[config.CLEAN_TEXT_COLUMN].str.strip() != ""].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.warning(
            f"Dropped {dropped} rows that became empty after preprocessing."
        )

    ensure_dir(config.PROCESSED_DIR)
    df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    logger.info(f"Processed data saved: {config.PROCESSED_DATA_PATH}")

    _preprocess_splits()
    return df


def _preprocess_splits() -> None:
    """Apply clean_text column to each saved split CSV."""
    for name, path in [
        ("train", config.TRAIN_PATH),
        ("val",   config.VAL_PATH),
        ("test",  config.TEST_PATH),
    ]:
        if not _is_valid_csv(path, min_rows=1):
            logger.warning(f"Split file missing or empty, skipping: {path}")
            continue

        split_df = pd.read_csv(path)

        # Skip if clean_text already present (e.g. partial previous run)
        if config.CLEAN_TEXT_COLUMN in split_df.columns:
            logger.info(f"{name} split already has clean_text — skipping.")
            continue

        split_df[config.CLEAN_TEXT_COLUMN] = preprocess_series(
            split_df[config.TEXT_COLUMN]
        )
        split_df = split_df[
            split_df[config.CLEAN_TEXT_COLUMN].str.strip() != ""
        ].reset_index(drop=True)
        split_df.to_csv(path, index=False)
        logger.info(f"Updated {name} split: {len(split_df):,} rows → {path}")


def _stage_eda(processed: pd.DataFrame) -> None:
    """Stage 4 — EDA. Always re-runs (fast, idempotent, no heavy computation)."""
    logger.info("[STAGE 4] Running EDA...")
    perform_eda(processed)


def _stage_features(processed: pd.DataFrame) -> tuple:
    """
    Stage 5 — Build TF-IDF feature matrices or load from cache.

    All five artifacts must be non-empty to count as a valid cache:
        X_train.npz, X_test.npz  (sparse matrices)
        y_train.csv, y_test.csv  (labels)
        feature_tfidf_vectorizer.pkl
    """
    features_valid = (
        _is_valid_npz(config.X_TRAIN_PATH)
        and _is_valid_npz(config.X_TEST_PATH)
        and _is_valid_csv(config.Y_TRAIN_PATH, min_rows=1)
        and _is_valid_csv(config.Y_TEST_PATH,  min_rows=1)
        and _is_valid_pkl(config.FEATURE_VEC_PATH)
    )
    if features_valid:
        logger.info("[CACHE] Feature matrices are valid — skipping feature engineering.")
        return load_features()

    logger.info("[STAGE 5] Running feature engineering (TF-IDF)...")
    X_train, X_test, y_train, y_test, vec = create_tfidf_features(processed)
    save_features(X_train, X_test, y_train, y_test, vec)
    return X_train, X_test, y_train, y_test, vec


def _stage_train_eval(
    X_train, X_test, y_train, y_test, tune: bool = False
) -> None:
    """Stage 6 — Train all models, evaluate each, compare, and save outputs."""
    logger.info(f"[STAGE 6] Training + Evaluation  (tune={tune})")

    trained_models = train_all_models(X_train, y_train, tune=tune)

    results = {}
    for name, model in trained_models.items():
        results[name] = evaluate_model(name, model, X_test, y_test)

    comparison = compare_models(results, trained_models)
    save_evaluation_outputs(results, comparison)

def _run_bert():
    """
    Run BERT training and evaluation using existing splits.
    """
    logger.info("[STAGE 6B] Running BERT training + evaluation...")

    from src.modeling.bert_trainer import train_bert
    from src.modeling.bert_evaluator import evaluate_bert
    from src.utils.helpers import get_label_encoder

    # Load splits
    train_df = pd.read_csv(config.TRAIN_PATH)
    val_df   = pd.read_csv(config.VAL_PATH)
    test_df  = pd.read_csv(config.TEST_PATH)

    label_encoder = get_label_encoder(config.CLASSES)

    # Train
    train_bert(train_df, val_df, label_encoder)

    # Evaluate
    metrics = evaluate_bert(test_df, label_encoder)

    logger.info(
        f"[BERT] Accuracy: {metrics['accuracy']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )


# ─────────────────────────────────────────────
# STAGE 7 — ADVANCED NLP ANALYSIS
# ─────────────────────────────────────────────

def run_advanced_nlp(df: pd.DataFrame = None) -> dict:
    """
    Execute Stage 7 — Advanced NLP Analysis.

    Can be called standalone (loads processed_data.csv automatically)
    or chained from run() by passing the processed DataFrame.

    Steps:
        7a. Sentiment analysis (VADER) → enriched_data.csv
        7b. Severity scoring           → enriched_data.csv (updated)
        7c. Topic modeling (LDA)       → topic_model_results.json + figures
        7d. Insights generation        → insights_report.md

    Args:
        df (pd.DataFrame | None): Processed DataFrame. If None, loads
                                  data/processed/processed_data.csv.

    Returns:
        dict with keys: enriched_df, topic_results, sentiment_stats,
                        severity_stats, insights_report
    """
    import json
    from src.analysis.sentiment_analyzer import analyze_sentiment, get_sentiment_stats
    from src.analysis.severity_scorer    import compute_severity, get_severity_stats
    from src.analysis.topic_model        import perform_topic_modeling, generate_insights

    logger.info("=" * 60)
    logger.info("MindScope — Stage 7: Advanced NLP Analysis")
    logger.info("=" * 60)

    if df is None:
        if not _is_valid_csv(config.PROCESSED_DATA_PATH, min_rows=1):
            raise FileNotFoundError(
                f"Processed data not found or empty: {config.PROCESSED_DATA_PATH}\n"
                "Run: python run_pipeline.py  first."
            )
        logger.info(f"Loading: {config.PROCESSED_DATA_PATH}")
        df = pd.read_csv(config.PROCESSED_DATA_PATH)

    # ── 7a: Sentiment ────────────────────────────────────────────────────
    enriched_valid = (
        _is_valid_csv(config.ENRICHED_DATA_PATH, min_rows=1)
        and "sentiment_score" in pd.read_csv(
            config.ENRICHED_DATA_PATH, nrows=0
        ).columns
    ) if _is_valid_csv(config.ENRICHED_DATA_PATH, min_rows=1) else False

    if enriched_valid:
        logger.info("[CACHE] enriched_data.csv with sentiment found.")
        enriched = pd.read_csv(config.ENRICHED_DATA_PATH)
    else:
        logger.info("[STAGE 7a] Sentiment analysis...")
        enriched = analyze_sentiment(df)

    sentiment_stats = get_sentiment_stats(enriched)

    # ── 7b: Severity ─────────────────────────────────────────────────────
    if "severity_score" not in enriched.columns:
        logger.info("[STAGE 7b] Severity scoring...")
        enriched = compute_severity(enriched)
    else:
        logger.info("[CACHE] severity_score column already present.")

    severity_stats = get_severity_stats(enriched)

    ensure_dir(config.PROCESSED_DIR)
    enriched.to_csv(config.ENRICHED_DATA_PATH, index=False)
    logger.info(f"Enriched data saved: {config.ENRICHED_DATA_PATH}")

    # ── 7c: Topic Modeling ───────────────────────────────────────────────
    if _is_valid_csv(config.TOPIC_RESULTS_PATH, min_rows=0) or \
       (os.path.exists(config.TOPIC_RESULTS_PATH) and
        os.path.getsize(config.TOPIC_RESULTS_PATH) > 0):
        logger.info("[CACHE] topic_model_results.json found.")
        with open(config.TOPIC_RESULTS_PATH) as f:
            topic_results = json.load(f)
    else:
        logger.info("[STAGE 7c] LDA topic modeling...")
        topic_results = perform_topic_modeling(enriched)

    # ── 7d: Insights ─────────────────────────────────────────────────────
    logger.info("[STAGE 7d] Generating insights report...")
    insights = generate_insights(
        enriched, topic_results, sentiment_stats, severity_stats
    )

    logger.info("=" * 60)
    logger.info("Stage 7 complete.")
    logger.info(f"  Enriched data   : {config.ENRICHED_DATA_PATH}")
    logger.info(f"  Insights report : {config.INSIGHTS_REPORT_PATH}")
    logger.info("=" * 60)

    return {
        "enriched_df":     enriched,
        "topic_results":   topic_results,
        "sentiment_stats": sentiment_stats,
        "severity_stats":  severity_stats,
        "insights_report": insights,
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindScope ML Pipeline")
    parser.add_argument(
    "--use_bert",
    action="store_true",
    help="Run BERT training and evaluation.",
    )

    parser.add_argument(
    "--skip_classical",
    action="store_true",
    help="Skip classical ML models and run only BERT.",
    )
    
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Skip the EDA stage.",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Enable GridSearchCV hyperparameter tuning.",
    )
    parser.add_argument(
        "--model-only", action="store_true",
        help="Skip stages 1-5; jump straight to training (features must exist).",
    )
    parser.add_argument(
        "--advanced", action="store_true",
        help="Run Stage 7 (advanced NLP: sentiment, severity, topics, insights).",
    )
    args = parser.parse_args()

    if args.advanced:
        run_advanced_nlp()
    else:
        run(
    skip_eda=args.skip_eda,
    tune=args.tune,
    model_only=args.model_only,
    use_bert=args.use_bert,
    skip_classical=args.skip_classical,
)