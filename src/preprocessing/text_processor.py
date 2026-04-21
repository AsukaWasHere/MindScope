"""
text_processor.py — NLP text preprocessing for the MindScope pipeline.

Responsibility (ONLY):
    Accept a raw text string and return a clean, normalized string
    ready for feature extraction (TF-IDF or embeddings).

Design decisions:
    - Lemmatization is chosen over stemming.
      Reason: Lemmatization produces real dictionary words ("running" → "run"),
      while stemming can produce non-words ("running" → "runn"). Since we use
      TF-IDF and transformer models that benefit from real vocabulary,
      lemmatization gives better feature quality at a small speed cost.

    - This module is STATELESS and PURE: the same input always returns the
      same output. No model state, no file I/O, no side effects.

    - spaCy is used for tokenization + lemmatization.
      NLTK is used for the English stopword list.
      Run setup: python -m spacy download en_core_web_sm

Input:  A single raw text string (combined title + body)
Output: A single cleaned string (tokens joined by spaces)
"""

import re

import nltk
import spacy
from nltk.corpus import stopwords

from src.utils import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# ONE-TIME RESOURCE LOADING
# These are loaded when the module is first imported, not on every call.
# ─────────────────────────────────────────────

def _load_resources():
    """
    Download NLTK stopwords if not already present, then load spaCy model.

    This runs once at import time. All subsequent calls to preprocess()
    reuse these in-memory objects.

    Returns:
        tuple: (spacy_nlp, stop_words_set)
    """
    # Download NLTK stopwords (skips download if already cached)
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words(config.STOPWORDS_LANG))

    # Load spaCy model (disable unused pipeline components for speed)
    try:
        nlp = spacy.load(config.SPACY_MODEL, disable=["parser", "ner"])
    except OSError:
        raise OSError(
            f"spaCy model '{config.SPACY_MODEL}' not found.\n"
            f"Run: python -m spacy download {config.SPACY_MODEL}"
        )

    logger.info(
        f"Text processor ready. "
        f"Stopwords: {len(stop_words)} | spaCy model: {config.SPACY_MODEL}"
    )
    return nlp, stop_words


# Load once at module import
_nlp, _stop_words = _load_resources()


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def preprocess(text: str) -> str:
    """
    Clean and normalize a single raw text string into a model-ready string.

    Processing steps (in order):
        1. Lowercase the entire string.
        2. Remove URLs (http/https/www patterns).
        3. Remove HTML tags and entities.
        4. Remove Reddit-specific artifacts (/r/ subreddit links, /u/ mentions).
        5. Remove all non-alphabetic characters (digits, punctuation, symbols).
        6. Collapse multiple spaces into one.
        7. Tokenize using spaCy.
        8. Remove NLTK English stopwords.
        9. Lemmatize each token using spaCy.
        10. Filter out single-character tokens.
        11. Re-join tokens into a clean string.

    Args:
        text (str): Raw combined text (title + body). May contain URLs,
                    HTML, Reddit jargon, and punctuation.

    Returns:
        str: Cleaned, lemmatized string with stopwords removed.
             Returns an empty string if input is None or non-string.

    Example:
        >>> preprocess("I've been struggling with /r/depression stuff... https://t.co/xyz")
        'struggling depression stuff'
    """
    # Guard: handle None or non-string input gracefully
    if not isinstance(text, str) or not text.strip():
        return ""

    text = _lowercase(text)
    text = _remove_urls(text)
    text = _remove_html(text)
    text = _remove_reddit_artifacts(text)
    text = _remove_non_alpha(text)
    text = _collapse_spaces(text)

    tokens = _tokenize_and_lemmatize(text)
    tokens = _remove_stopwords(tokens)
    tokens = _remove_short_tokens(tokens)

    return " ".join(tokens)


def preprocess_series(series) -> "pd.Series":
    """
    Apply preprocess() to every element of a pandas Series.

    A convenience wrapper used by tfidf_vectorizer.py and embedder.py
    to process the entire `text` column of a DataFrame at once.

    Args:
        series (pd.Series): Series of raw text strings.

    Returns:
        pd.Series: Series of cleaned strings (same index as input).

    Example:
        >>> clean = preprocess_series(df["text"])
    """
    logger.info(f"Preprocessing {len(series)} text samples...")
    cleaned = series.apply(preprocess)
    empty_count = (cleaned == "").sum()
    if empty_count > 0:
        logger.warning(f"{empty_count} samples became empty after preprocessing.")
    return cleaned


# ─────────────────────────────────────────────
# PRIVATE STEP FUNCTIONS
# Each function does exactly one thing.
# ─────────────────────────────────────────────

def _lowercase(text: str) -> str:
    """Convert all characters to lowercase."""
    return text.lower()


def _remove_urls(text: str) -> str:
    """
    Remove http://, https://, and www. URLs from text.

    Pattern covers:
        - https://example.com/path?query=1
        - http://t.co/abc123
        - www.reddit.com/r/something
    """
    url_pattern = r"http\S+|https\S+|www\.\S+"
    return re.sub(url_pattern, " ", text)


def _remove_html(text: str) -> str:
    """
    Remove HTML tags (e.g., <br>, <p>) and common HTML entities
    (e.g., &amp;, &lt;, &gt;, &nbsp;).
    """
    text = re.sub(r"<[^>]+>", " ", text)           # tags: <br />, <p class=...>
    text = re.sub(r"&[a-z]+;", " ", text)           # entities: &amp; &lt;
    return text


def _remove_reddit_artifacts(text: str) -> str:
    """
    Remove Reddit-specific formatting artifacts.

    Removes:
        - Subreddit links: /r/depression, r/adhd
        - User mentions: /u/username, u/username
        - Markdown bold/italic: **text**, *text*
    """
    text = re.sub(r"/?r/\w+", " ", text)            # subreddit links
    text = re.sub(r"/?u/\w+", " ", text)            # user mentions
    text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)  # markdown bold/italic
    return text


def _remove_non_alpha(text: str) -> str:
    """
    Keep only lowercase alphabetic characters and spaces.

    This removes punctuation, digits, and symbols. We don't need
    numbers for mental health condition classification (e.g., "3 days"
    becomes "days" which is fine).
    """
    return re.sub(r"[^a-z\s]", " ", text)


def _collapse_spaces(text: str) -> str:
    """
    Replace multiple consecutive spaces with a single space and strip ends.
    """
    return re.sub(r"\s+", " ", text).strip()


def _tokenize_and_lemmatize(text: str) -> list:
    """
    Use spaCy to tokenize the text and lemmatize each token.

    spaCy's lemmatizer maps:
        "running" → "run"
        "better"  → "good"
        "children" → "child"

    The `_nlp` object is loaded once at module import and reused here.

    Args:
        text (str): Cleaned, lowercased text.

    Returns:
        list: List of lemmatized token strings.
    """
    doc = _nlp(text)
    return [token.lemma_ for token in doc]


def _remove_stopwords(tokens: list) -> list:
    """
    Remove common English stopwords from the token list.

    Stopwords ("the", "is", "at", "which", ...) appear frequently
    across all classes and add noise without discriminating power.

    Args:
        tokens (list): List of token strings.

    Returns:
        list: Filtered list with stopwords removed.
    """
    return [t for t in tokens if t not in _stop_words]


def _remove_short_tokens(tokens: list, min_length: int = 2) -> list:
    """
    Remove tokens that are too short to carry meaning.

    Single characters ("a", "i") slip through stopword removal and
    add noise. Minimum length of 2 keeps useful short words like
    "no", "ok", "go" while filtering out isolated letters.

    Args:
        tokens (list): List of token strings.
        min_length (int): Minimum character length to keep. Default: 2.

    Returns:
        list: Filtered list with short tokens removed.
    """
    return [t for t in tokens if len(t) >= min_length]