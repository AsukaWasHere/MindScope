"""
loader.py — Data ingestion module for the MindScope pipeline.

Responsibility (ONLY):
    Load raw CSV files, assign class labels, merge into one DataFrame,
    and save to data/processed/merged_raw.csv.

This module does NOT clean, preprocess, or split data.
That is the job of cleaner.py.

Input:  data/raw/*.csv  (five files, one per subreddit)
Output: data/processed/merged_raw.csv
"""

import os

import pandas as pd

from src.utils import config
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load all five raw CSV files, assign labels, merge, and save to disk.

    Steps:
        1. Read each CSV from data/raw/ using the paths in config.RAW_FILES.
        2. Add a `label` column to each DataFrame (value = the dict key,
           e.g., "adhd", "depression", etc.).
        3. Keep only the columns listed in config.KEEP_COLUMNS + label.
        4. Concatenate all five DataFrames into one.
        5. Reset the index so it is contiguous (0, 1, 2, …).
        6. Save the merged DataFrame to config.MERGED_RAW_PATH.
        7. Log class distribution so any imbalance is visible immediately.

    Returns:
        pd.DataFrame: The merged DataFrame with columns:
                      body, title, subreddit, score, num_comments, label

    Raises:
        FileNotFoundError: If any of the raw CSV files are missing.

    Example:
        >>> from src.ingestion.loader import load_data
        >>> df = load_data()
        >>> df.shape
        (50000, 6)          # exact shape depends on your data
        >>> df["label"].value_counts()
        depression    12000
        adhd          10000
        ...
    """
    logger.info("=== Stage 1: Data Ingestion ===")

    frames = []

    for label, filepath in config.RAW_FILES.items():
        _check_file_exists(filepath)
        df = _read_single_csv(filepath, label)
        frames.append(df)

    merged = _merge_frames(frames)
    _save_merged(merged)
    _log_class_distribution(merged)

    logger.info(f"Ingestion complete. Total rows: {len(merged)}")
    return merged


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _check_file_exists(filepath: str) -> None:
    """
    Raise FileNotFoundError if the given path does not exist.

    Args:
        filepath (str): Absolute path to the CSV file.

    Raises:
        FileNotFoundError: If the file is not found at the given path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Raw data file not found: {filepath}\n"
            f"Place all five CSV files in: {config.RAW_DIR}"
        )


def _read_single_csv(filepath: str, label: str) -> pd.DataFrame:
    """
    Read one CSV file, add the label column, and keep only required columns.

    Args:
        filepath (str): Path to the CSV file.
        label (str):    Class label string (e.g., "adhd", "depression").

    Returns:
        pd.DataFrame: DataFrame with columns from config.KEEP_COLUMNS + label.
    """
    logger.info(f"Loading: {os.path.basename(filepath)}")
    df = pd.read_csv(filepath, low_memory=False)

    # Assign class label BEFORE selecting columns
    df[config.LABEL_COLUMN] = label

    # Keep only the columns we actually need; drop everything else
    columns_to_keep = config.KEEP_COLUMNS + [config.LABEL_COLUMN]

    # Safety: only select columns that actually exist in this CSV
    available = [col for col in columns_to_keep if col in df.columns]
    missing = set(columns_to_keep) - set(available)
    if missing:
        logger.warning(f"Columns missing in {os.path.basename(filepath)}: {missing}")

    df = df[available]
    logger.info(f"  → {len(df)} rows loaded for label='{label}'")
    return df


def _merge_frames(frames: list) -> pd.DataFrame:
    """
    Concatenate a list of DataFrames into one and reset the index.

    Args:
        frames (list): List of pd.DataFrame objects, one per subreddit.

    Returns:
        pd.DataFrame: Single merged DataFrame with a fresh integer index.
    """
    logger.info("Merging all DataFrames...")
    merged = pd.concat(frames, axis=0, ignore_index=True)
    return merged


def _save_merged(df: pd.DataFrame) -> None:
    """
    Save the merged DataFrame to data/processed/merged_raw.csv.

    Creates the output directory if it does not exist.

    Args:
        df (pd.DataFrame): Merged DataFrame to persist.
    """
    ensure_dir(config.PROCESSED_DIR)
    df.to_csv(config.MERGED_RAW_PATH, index=False)
    logger.info(f"Merged data saved to: {config.MERGED_RAW_PATH}")


def _log_class_distribution(df: pd.DataFrame) -> None:
    """
    Log the number of posts per class label.

    Useful for immediately spotting class imbalance after loading.

    Args:
        df (pd.DataFrame): Merged DataFrame that has a `label` column.
    """
    counts = df[config.LABEL_COLUMN].value_counts()
    logger.info("Class distribution:")
    for label, count in counts.items():
        pct = 100 * count / len(df)
        logger.info(f"  {label:<12} {count:>6} rows  ({pct:.1f}%)")