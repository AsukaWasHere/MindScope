"""
tfidf_vectorizer.py — TF-IDF feature engineering for the MindScope pipeline.

Responsibility (ONLY):
    Convert `clean_text` into a sparse TF-IDF feature matrix, split into
    train/test, and save all artifacts (matrices, labels, vectorizer).

Key rules:
    - Vectorizer is fit on X_train ONLY — prevents data leakage.
    - Sparse .npz format for matrices — far more efficient than dense CSV.
    - All paths and hyperparameters come from config.py — no hardcoding.

Public API:
    create_tfidf_features(df)  → (X_train, X_test, y_train, y_test, vectorizer)
    split_data(df)             → (train_df, test_df)
    save_features(...)         → writes all artifacts to disk
    load_features()            → reloads all artifacts from disk

Input:  pd.DataFrame with `clean_text` and `label` columns
Output:
    data/features/X_train.npz
    data/features/X_test.npz
    data/features/y_train.csv
    data/features/y_test.csv
    models/saved/feature_tfidf_vectorizer.pkl
"""

import scipy.sparse as sp
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.utils import config
from src.utils.helpers import ensure_dir, save_object
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def create_tfidf_features(df: pd.DataFrame) -> tuple:
    """
    Full TF-IDF feature engineering pipeline — split, fit, transform.

    Steps:
        1. Validate required columns exist.
        2. 80/20 stratified split → train_df, test_df.
        3. Fit TfidfVectorizer on train text only.
        4. Transform both splits into sparse matrices.
        5. Log vocabulary size and matrix shapes.

    Args:
        df (pd.DataFrame): Must have `clean_text` and `label` columns.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, vectorizer)
            X_train    — scipy.sparse.csr_matrix  shape (n_train, max_features)
            X_test     — scipy.sparse.csr_matrix  shape (n_test,  max_features)
            y_train    — pd.Series of string labels
            y_test     — pd.Series of string labels
            vectorizer — fitted TfidfVectorizer

    Example:
        >>> df = pd.read_csv("data/processed/processed_data.csv")
        >>> X_tr, X_te, y_tr, y_te, vec = create_tfidf_features(df)
        >>> X_tr.shape
        (40000, 5000)
    """
    _validate_columns(df)
    logger.info("=== Feature Engineering: TF-IDF ===")

    train_df, test_df = split_data(df)

    X_train_text = train_df[config.CLEAN_TEXT_COLUMN]
    X_test_text  = test_df[config.CLEAN_TEXT_COLUMN]
    y_train      = train_df[config.LABEL_COLUMN].reset_index(drop=True)
    y_test       = test_df[config.LABEL_COLUMN].reset_index(drop=True)

    vectorizer = _build_vectorizer()

    logger.info(f"Fitting vectorizer on {len(X_train_text):,} training samples...")
    X_train = vectorizer.fit_transform(X_train_text)

    logger.info(f"Transforming {len(X_test_text):,} test samples...")
    X_test = vectorizer.transform(X_test_text)

    logger.info(f"Vocabulary size : {len(vectorizer.vocabulary_):,} terms")
    logger.info(f"X_train shape   : {X_train.shape}")
    logger.info(f"X_test  shape   : {X_test.shape}")
    logger.info(f"y_train dist    : {y_train.value_counts().to_dict()}")
    logger.info(f"y_test  dist    : {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test, vectorizer


def split_data(df: pd.DataFrame) -> tuple:
    """
    Stratified 80/20 train/test split on the processed DataFrame.

    Stratification on `label` guarantees proportional class balance.
    Ratios come from config.FEATURE_TRAIN_RATIO / FEATURE_TEST_RATIO.

    Note: This is a separate split from the 70/15/15 pipeline split in
    cleaner.py. The feature matrices use their own 80/20 split.

    Args:
        df (pd.DataFrame): Processed DataFrame with `label` column.

    Returns:
        tuple: (train_df, test_df) — both with reset index.

    Example:
        >>> train, test = split_data(df)
    """
    logger.info(
        f"Splitting: {config.FEATURE_TRAIN_RATIO:.0%} train / "
        f"{config.FEATURE_TEST_RATIO:.0%} test  (stratified by label)"
    )
    train_df, test_df = train_test_split(
        df,
        test_size=config.FEATURE_TEST_RATIO,
        stratify=df[config.LABEL_COLUMN],
        random_state=config.RANDOM_SEED,
    )
    logger.info(f"  Train : {len(train_df):,} rows")
    logger.info(f"  Test  : {len(test_df):,} rows")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_features(
    X_train,
    X_test,
    y_train: pd.Series,
    y_test:  pd.Series,
    vectorizer: TfidfVectorizer,
) -> None:
    """
    Persist all feature engineering artifacts to disk.

    Saved artifacts:
        data/features/X_train.npz                    — sparse train matrix
        data/features/X_test.npz                     — sparse test matrix
        data/features/y_train.csv                    — train labels (CSV)
        data/features/y_test.csv                     — test labels (CSV)
        models/saved/feature_tfidf_vectorizer.pkl    — fitted vectorizer

    Args:
        X_train: scipy.sparse matrix — training features.
        X_test:  scipy.sparse matrix — test features.
        y_train (pd.Series): Training class labels.
        y_test  (pd.Series): Test class labels.
        vectorizer (TfidfVectorizer): Fitted vectorizer to pickle.

    Example:
        >>> save_features(X_train, X_test, y_train, y_test, vec)
    """
    ensure_dir(config.FEATURES_DIR)
    ensure_dir(config.MODELS_DIR)

    sp.save_npz(config.X_TRAIN_PATH, X_train)
    logger.info(f"X_train saved : {config.X_TRAIN_PATH}  {X_train.shape}")

    sp.save_npz(config.X_TEST_PATH, X_test)
    logger.info(f"X_test  saved : {config.X_TEST_PATH}   {X_test.shape}")

    pd.DataFrame({config.LABEL_COLUMN: y_train}).to_csv(
        config.Y_TRAIN_PATH, index=False
    )
    logger.info(f"y_train saved : {config.Y_TRAIN_PATH}")

    pd.DataFrame({config.LABEL_COLUMN: y_test}).to_csv(
        config.Y_TEST_PATH, index=False
    )
    logger.info(f"y_test  saved : {config.Y_TEST_PATH}")

    save_object(vectorizer, config.FEATURE_VEC_PATH)
    logger.info(f"Vectorizer    : {config.FEATURE_VEC_PATH}")


def load_features() -> tuple:
    """
    Reload all saved feature artifacts from disk.

    Used by trainer.py and evaluator.py to skip feature re-computation
    when the artifacts already exist.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, vectorizer)

    Raises:
        FileNotFoundError: Via helpers.load_object if any file is missing.

    Example:
        >>> X_tr, X_te, y_tr, y_te, vec = load_features()
    """
    from src.utils.helpers import load_object

    logger.info("Loading pre-computed TF-IDF features from disk...")
    X_train    = sp.load_npz(config.X_TRAIN_PATH)
    X_test     = sp.load_npz(config.X_TEST_PATH)
    y_train    = pd.read_csv(config.Y_TRAIN_PATH)[config.LABEL_COLUMN]
    y_test     = pd.read_csv(config.Y_TEST_PATH)[config.LABEL_COLUMN]
    vectorizer = load_object(config.FEATURE_VEC_PATH)

    logger.info(f"X_train : {X_train.shape}  |  X_test : {X_test.shape}")
    return X_train, X_test, y_train, y_test, vectorizer


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are absent."""
    required = {config.CLEAN_TEXT_COLUMN, config.LABEL_COLUMN}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"TF-IDF step requires columns {required}. Missing: {missing}\n"
            "Ensure processed_data.csv contains a `clean_text` column."
        )


def _build_vectorizer() -> TfidfVectorizer:
    """
    Construct a TfidfVectorizer using hyperparameters from config.

    Key choices:
        sublinear_tf=True  — applies log(1+tf) to compress high-frequency
                             term dominance; consistently improves text clf.
        min_df=2           — ignores hapax legomena (terms in < 2 docs)
                             that add noise without generalisable signal.
        token_pattern      — only keeps lowercase alphabetic tokens of
                             length >= 2 (consistent with text_processor.py).

    Returns:
        TfidfVectorizer: Unfitted instance.
    """
    vec = TfidfVectorizer(
        max_features=config.FEATURE_TFIDF_MAX_FEATURES,
        ngram_range=config.FEATURE_TFIDF_NGRAM_RANGE,
        sublinear_tf=config.FEATURE_TFIDF_SUBLINEAR_TF,
        min_df=config.FEATURE_TFIDF_MIN_DF,
        token_pattern=r"(?u)\b[a-z]{2,}\b",
        strip_accents="unicode",
    )
    logger.info(
        f"TfidfVectorizer: max_features={config.FEATURE_TFIDF_MAX_FEATURES}, "
        f"ngram_range={config.FEATURE_TFIDF_NGRAM_RANGE}, "
        f"sublinear_tf={config.FEATURE_TFIDF_SUBLINEAR_TF}, "
        f"min_df={config.FEATURE_TFIDF_MIN_DF}"
    )
    return vec