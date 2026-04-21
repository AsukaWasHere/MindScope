"""
cleaner.py — Data cleaning module for the MindScope pipeline.

Responsibility (ONLY):
    Accept the merged raw DataFrame, apply structural cleaning rules,
    combine title + body into a `text` column, split into train/val/test,
    and save all outputs to disk.

This module does NOT do NLP preprocessing (no tokenization, no stopword
removal). That is the job of text_processor.py.

Input:  data/processed/merged_raw.csv
Output:
    data/processed/cleaned.csv
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import config
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning pipeline on the merged raw DataFrame.

    Cleaning steps applied in order:
        1. Drop rows where `body` is null, "[deleted]", or "[removed]".
        2. Drop rows where `title` is null or empty.
        3. Remove duplicate rows (by exact body+title match, since we
           don't have `id` after column selection in loader).
        4. Filter out posts where `body` has fewer than MIN_BODY_WORD_COUNT words.
        5. Combine `title` and `body` into a single `text` column.
        6. Strip leading/trailing whitespace from `text`.

    Args:
        df (pd.DataFrame): The merged raw DataFrame from loader.load_data().

    Returns:
        pd.DataFrame: Cleaned DataFrame with an added `text` column.

    Example:
        >>> from src.ingestion.loader import load_data
        >>> from src.preprocessing.cleaner import clean_data
        >>> raw = load_data()
        >>> cleaned = clean_data(raw)
        >>> "text" in cleaned.columns
        True
    """
    logger.info("=== Stage 2: Data Cleaning ===")
    logger.info(f"Input rows: {len(df)}")

    df = _drop_deleted_bodies(df)
    df = _drop_empty_titles(df)
    df = _drop_duplicates(df)
    df = _drop_short_posts(df)
    df = _combine_text_columns(df)

    logger.info(f"Rows after cleaning: {len(df)}")
    return df


def save_cleaned_data(df: pd.DataFrame) -> None:
    """
    Persist the cleaned DataFrame and create stratified train/val/test splits.

    Saves:
        - data/processed/cleaned.csv  — full cleaned dataset
        - data/splits/train.csv       — 70% stratified by label
        - data/splits/val.csv         — 15% stratified by label
        - data/splits/test.csv        — 15% stratified by label

    The splits are saved once and reused on subsequent runs. They are
    NOT regenerated if the files already exist (see run_pipeline.py).

    Args:
        df (pd.DataFrame): Cleaned DataFrame (output of clean_data()).

    Example:
        >>> save_cleaned_data(cleaned_df)
        # Writes four CSV files to disk.
    """
    ensure_dir(config.PROCESSED_DIR)
    ensure_dir(config.SPLITS_DIR)

    # Save full cleaned dataset
    df.to_csv(config.CLEANED_DATA_PATH, index=False)
    logger.info(f"Cleaned data saved to: {config.CLEANED_DATA_PATH}")

    # Create and save train/val/test splits
    train, val, test = _split_data(df)

    train.to_csv(config.TRAIN_PATH, index=False)
    val.to_csv(config.VAL_PATH, index=False)
    test.to_csv(config.TEST_PATH, index=False)

    logger.info(f"Train split saved: {len(train)} rows → {config.TRAIN_PATH}")
    logger.info(f"Val   split saved: {len(val)}   rows → {config.VAL_PATH}")
    logger.info(f"Test  split saved: {len(test)}  rows → {config.TEST_PATH}")


# ─────────────────────────────────────────────
# PRIVATE CLEANING STEPS
# ─────────────────────────────────────────────

def _drop_deleted_bodies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where `body` is null or contains Reddit deletion markers.

    Reddit replaces removed/deleted post content with the literal strings
    "[deleted]" or "[removed]". These carry no signal for classification.

    Args:
        df (pd.DataFrame): DataFrame with a `body` column.

    Returns:
        pd.DataFrame: DataFrame with deleted/null body rows removed.
    """
    before = len(df)

    # Drop rows with a null body
    df = df[df["body"].notna()]

    # Drop rows where body is a deletion marker (case-insensitive strip)
    mask = df["body"].str.strip().isin(config.DELETED_MARKERS)
    df = df[~mask]

    logger.info(f"Dropped {before - len(df)} rows with deleted/null body.")
    return df.reset_index(drop=True)


def _drop_empty_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where `title` is null or contains only whitespace.

    The title is a key signal (headline intent), so posts missing it
    are dropped rather than imputed.

    Args:
        df (pd.DataFrame): DataFrame with a `title` column.

    Returns:
        pd.DataFrame: DataFrame with empty-title rows removed.
    """
    before = len(df)
    df = df[df["title"].notna()]
    df = df[df["title"].str.strip() != ""]
    logger.info(f"Dropped {before - len(df)} rows with empty title.")
    return df.reset_index(drop=True)


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate posts based on the body + title combination.

    Cross-posts and reposts produce identical body+title pairs and should
    be deduplicated to prevent data leakage between splits.

    Args:
        df (pd.DataFrame): DataFrame with `body` and `title` columns.

    Returns:
        pd.DataFrame: DataFrame with duplicate rows removed (first kept).
    """
    before = len(df)
    df = df.drop_duplicates(subset=["body", "title"], keep="first")
    logger.info(f"Dropped {before - len(df)} duplicate rows.")
    return df.reset_index(drop=True)


def _drop_short_posts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove posts where the `body` contains fewer than MIN_BODY_WORD_COUNT words.

    Very short bodies (e.g., "Help", "Same") provide no useful signal for
    multi-class classification and are more likely to be misclassified.

    Args:
        df (pd.DataFrame): DataFrame with a `body` column.

    Returns:
        pd.DataFrame: DataFrame with short-body rows removed.
    """
    before = len(df)
    word_counts = df["body"].str.split().str.len()
    df = df[word_counts >= config.MIN_BODY_WORD_COUNT]
    logger.info(
        f"Dropped {before - len(df)} rows with body < "
        f"{config.MIN_BODY_WORD_COUNT} words."
    )
    return df.reset_index(drop=True)


def _combine_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate `title` and `body` into a single `text` column.

    The combined field is what all downstream modules (text_processor,
    vectorizer, embedder) operate on. Combining captures both the
    post headline and the full narrative in one feature.

    Format: "<title> <body>"

    Args:
        df (pd.DataFrame): DataFrame with `title` and `body` columns.

    Returns:
        pd.DataFrame: DataFrame with an added `text` column.
    """
    df[config.TEXT_COLUMN] = (
        df["title"].str.strip() + " " + df["body"].str.strip()
    )
    logger.info("Created `text` column: title + body combined.")
    return df


# ─────────────────────────────────────────────
# PRIVATE SPLIT HELPER
# ─────────────────────────────────────────────

def _split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train / val / test splits.

    Stratification on `label` ensures each split has proportional
    representation of all five classes — important when class sizes differ.

    Split ratios come from config:
        TRAIN_RATIO = 0.70
        VAL_RATIO   = 0.15
        TEST_RATIO  = 0.15

    The two-step approach:
        Step 1: Split into train (70%) and temp (30%).
        Step 2: Split temp into val (50% of 30% = 15%) and test (50% of 30% = 15%).

    Args:
        df (pd.DataFrame): Full cleaned DataFrame.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    logger.info(
        f"Splitting data: {config.TRAIN_RATIO}/{config.VAL_RATIO}/{config.TEST_RATIO}"
    )

    # Step 1: train vs. temp
    train, temp = train_test_split(
        df,
        test_size=(1 - config.TRAIN_RATIO),
        stratify=df[config.LABEL_COLUMN],
        random_state=config.RANDOM_SEED,
    )

    # Step 2: val vs. test (equal halves of temp)
    val, test = train_test_split(
        temp,
        test_size=0.5,
        stratify=temp[config.LABEL_COLUMN],
        random_state=config.RANDOM_SEED,
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )