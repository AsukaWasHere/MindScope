"""
config.py — Central configuration for the MindScope pipeline.

All constants, file paths, hyperparameters, and settings live here.
Every other module imports from this file. This file imports nothing
from the rest of the project.
"""

import os

# ─────────────────────────────────────────────
# ROOT PATHS
# ─────────────────────────────────────────────

# Project root is two levels above this file: src/utils/ → src/ → root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR        = os.path.join(ROOT_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
SPLITS_DIR      = os.path.join(DATA_DIR, "splits")
MODELS_DIR      = os.path.join(ROOT_DIR, "models", "saved")
REPORTS_DIR     = os.path.join(ROOT_DIR, "reports")
FIGURES_DIR     = os.path.join(REPORTS_DIR, "figures")
METRICS_DIR     = os.path.join(REPORTS_DIR, "metrics")

# ─────────────────────────────────────────────
# RAW DATA FILES
# ─────────────────────────────────────────────

# Maps filename stem → class label string
RAW_FILES = {
    "adhd":       os.path.join(RAW_DIR, "adhd.csv"),
    "aspergers":  os.path.join(RAW_DIR, "aspergers.csv"),
    "depression": os.path.join(RAW_DIR, "depression.csv"),
    "ocd":        os.path.join(RAW_DIR, "ocd.csv"),
    "ptsd":       os.path.join(RAW_DIR, "ptsd.csv"),
}

# ─────────────────────────────────────────────
# PROCESSED FILE PATHS
# ─────────────────────────────────────────────

MERGED_RAW_PATH     = os.path.join(PROCESSED_DIR, "merged_raw.csv")
CLEANED_DATA_PATH   = os.path.join(PROCESSED_DIR, "cleaned.csv")

TRAIN_PATH  = os.path.join(SPLITS_DIR, "train.csv")
VAL_PATH    = os.path.join(SPLITS_DIR, "val.csv")
TEST_PATH   = os.path.join(SPLITS_DIR, "test.csv")

# ─────────────────────────────────────────────
# COLUMNS TO KEEP AFTER LOADING
# ─────────────────────────────────────────────

# These are the only columns retained from the raw CSVs
KEEP_COLUMNS = ["body", "title", "subreddit", "score", "num_comments"]

# Column added during ingestion
LABEL_COLUMN = "label"

# Combined text column created during cleaning
TEXT_COLUMN = "text"

# Final preprocessed text column produced by text_processor
CLEAN_TEXT_COLUMN = "clean_text"

# ─────────────────────────────────────────────
# CLEANING RULES
# ─────────────────────────────────────────────

# Placeholder strings Reddit uses for deleted/removed content
DELETED_MARKERS = ["[deleted]", "[removed]"]

# Posts with fewer words than this in `body` are dropped
MIN_BODY_WORD_COUNT = 10

# ─────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT RATIOS
# ─────────────────────────────────────────────

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15   # Implicit: 1 - TRAIN_RATIO - VAL_RATIO

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────

RANDOM_SEED = 42

# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────

# NLTK stopwords language
STOPWORDS_LANG = "english"

# spaCy model used for lemmatization (must be downloaded separately)
# Run: python -m spacy download en_core_web_sm
SPACY_MODEL = "en_core_web_sm"

# ─────────────────────────────────────────────
# TFIDF VECTORIZER
# ─────────────────────────────────────────────

TFIDF_MAX_FEATURES  = 50000
TFIDF_NGRAM_RANGE   = (1, 2)
TFIDF_SUBLINEAR_TF  = True

TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

# ─────────────────────────────────────────────
# TRANSFORMER EMBEDDINGS
# ─────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────
# BERT CONFIGURATION
# ─────────────────────────────────────────────

BERT_CONFIG = {
    "model_name": "bert-base-uncased",
    "save_dir": os.path.join(MODELS_DIR, "bert"),
    "max_length": 128,
    "epochs": 1,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "log_every_n_steps": 50,
}

# ─────────────────────────────────────────────
# CLASS LABELS (ordered consistently)
# ─────────────────────────────────────────────

CLASSES = ["adhd", "aspergers", "depression", "ocd", "ptsd"]

# Per-class color palette — used consistently across all EDA plots
CLASS_COLORS = {
    "adhd":       "#E07B54",   # warm orange
    "aspergers":  "#5B8DB8",   # steel blue
    "depression": "#7A6BAE",   # muted violet
    "ocd":        "#4FAF8C",   # teal green
    "ptsd":       "#C0575A",   # dusty rose
}

# ─────────────────────────────────────────────
# EDA SETTINGS
# ─────────────────────────────────────────────

# Top-N most frequent words shown per class in word frequency analysis
EDA_TOP_N_WORDS = 20

# Word cloud output size (width × height in pixels)
WORDCLOUD_WIDTH  = 800
WORDCLOUD_HEIGHT = 400

# Maximum words rendered in each word cloud
WORDCLOUD_MAX_WORDS = 100

# ─────────────────────────────────────────────
# FEATURE ENGINEERING — TF-IDF (task-level override)
# ─────────────────────────────────────────────
# NOTE: The pipeline-level vectorizer (max_features=50000) lives above.
# For the EDA/feature step we use a smaller, faster setting
# so the saved X_train/X_test matrices stay manageable on disk.

FEATURE_TFIDF_MAX_FEATURES = 5000
FEATURE_TFIDF_NGRAM_RANGE  = (1, 2)
FEATURE_TFIDF_SUBLINEAR_TF = True
FEATURE_TFIDF_MIN_DF       = 2        # ignore terms that appear in < 2 docs

# ─────────────────────────────────────────────
# FEATURE ENGINEERING — TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
# NOTE: The pipeline uses a 70/15/15 split saved to data/splits/.
# The task prompt requests an 80/20 split for the feature matrices
# (X_train, X_test). These are separate artifacts saved to features/.

FEATURE_TRAIN_RATIO = 0.80
FEATURE_TEST_RATIO  = 0.20

# ─────────────────────────────────────────────
# FEATURE ENGINEERING — OUTPUT PATHS
# ─────────────────────────────────────────────

FEATURES_DIR = os.path.join(DATA_DIR, "features")

X_TRAIN_PATH      = os.path.join(FEATURES_DIR, "X_train.npz")
X_TEST_PATH       = os.path.join(FEATURES_DIR, "X_test.npz")
Y_TRAIN_PATH      = os.path.join(FEATURES_DIR, "y_train.csv")
Y_TEST_PATH       = os.path.join(FEATURES_DIR, "y_test.csv")
FEATURE_VEC_PATH  = os.path.join(MODELS_DIR,   "feature_tfidf_vectorizer.pkl")

# ─────────────────────────────────────────────
# PROCESSED DATA PATH (output of run_pipeline Stage 3)
# ─────────────────────────────────────────────

PROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_data.csv")

# ─────────────────────────────────────────────
# PIPELINE CONTROL
# ─────────────────────────────────────────────

PIPELINE_CONFIG = {
    "run_logreg": True,
    "run_random_forest": True,
    "run_lightgbm": True,
    "use_bert": False,
}

# ─────────────────────────────────────────────
# MODELING — MODEL REGISTRY
# ─────────────────────────────────────────────

# All supported model keys — used as filenames and dict keys throughout
MODEL_NAMES = ["logistic_regression", "naive_bayes", "svm"]

# Saved artifact paths — one pkl per model
MODEL_PATHS = {
    name: os.path.join(MODELS_DIR, f"{name}.pkl")
    for name in MODEL_NAMES
}

# Best model artifact (written by trainer after comparison)
BEST_MODEL_PATH        = os.path.join(MODELS_DIR, "best_model.pkl")
BEST_MODEL_NAME_PATH   = os.path.join(MODELS_DIR, "best_model_name.txt")
LABEL_ENCODER_PATH     = os.path.join(MODELS_DIR, "label_encoder.pkl")

# ─────────────────────────────────────────────
# MODELING — HYPERPARAMETERS
# ─────────────────────────────────────────────

# Logistic Regression
LR_C            = 5.0          # inverse regularisation strength (higher = less reg)
LR_MAX_ITER     = 1000
LR_SOLVER       = "lbfgs"
LR_MULTI_CLASS  = "auto"

# Naive Bayes — MultinomialNB
# alpha: Laplace smoothing (0 = no smoothing, 1 = full Laplace)
NB_ALPHA        = 0.1

# SVM — LinearSVC
SVM_C           = 1.0
SVM_MAX_ITER    = 2000

# ─────────────────────────────────────────────
# MODELING — HYPERPARAMETER TUNING
# ─────────────────────────────────────────────

# GridSearchCV folds
GRIDSEARCH_CV   = 3
GRIDSEARCH_SCORING = "f1_macro"

GRIDSEARCH_PARAMS = {
    "logistic_regression": {
        "C":        [0.1, 1.0, 5.0, 10.0],
        "solver":   ["lbfgs", "saga"],
    },
    "naive_bayes": {
        "alpha":    [0.01, 0.1, 0.5, 1.0],
    },
    "svm": {
        "C":        [0.1, 0.5, 1.0, 5.0],
    },
}

# ─────────────────────────────────────────────
# EVALUATION — OUTPUT PATHS
# ─────────────────────────────────────────────

ALL_METRICS_PATH       = os.path.join(METRICS_DIR, "all_models_metrics.json")
COMPARISON_TABLE_PATH  = os.path.join(METRICS_DIR, "model_comparison.csv")
CONFUSION_MATRIX_PATH  = os.path.join(FIGURES_DIR, "confusion_matrix_{name}.png")
METRICS_REPORT_PATH    = os.path.join(METRICS_DIR, "{name}_classification_report.txt")

# ─────────────────────────────────────────────
# ADVANCED NLP ANALYSIS — PATHS
# ─────────────────────────────────────────────

# Enriched dataset with sentiment + severity columns added
ENRICHED_DATA_PATH = os.path.join(PROCESSED_DIR, "enriched_data.csv")

# Sentiment analysis outputs
SENTIMENT_STATS_PATH  = os.path.join(METRICS_DIR, "sentiment_stats.json")
SENTIMENT_FIGURE_PATH = os.path.join(FIGURES_DIR, "sentiment_distribution.png")

# Severity outputs
SEVERITY_STATS_PATH  = os.path.join(METRICS_DIR, "severity_stats.json")
SEVERITY_FIGURE_PATH = os.path.join(FIGURES_DIR, "severity_distribution.png")

# Topic modeling outputs
TOPIC_MODEL_PATH      = os.path.join(MODELS_DIR, "lda_model.pkl")
TOPIC_RESULTS_PATH    = os.path.join(METRICS_DIR, "topic_model_results.json")
TOPIC_FIGURE_PATH     = os.path.join(FIGURES_DIR, "topic_keywords.png")

# Insights report (human-readable markdown)
INSIGHTS_REPORT_PATH  = os.path.join(METRICS_DIR, "insights_report.md")

# ─────────────────────────────────────────────
# ADVANCED NLP ANALYSIS — SETTINGS
# ─────────────────────────────────────────────

# VADER compound score thresholds for label assignment
VADER_POSITIVE_THRESH =  0.05   # compound >=  0.05 → "positive"
VADER_NEGATIVE_THRESH = -0.05   # compound <= -0.05 → "negative"
                                 # else              → "neutral"

# Severity scoring weights (must sum to 1.0)
SEVERITY_SENTIMENT_WEIGHT  = 0.40   # contribution of VADER negativity
SEVERITY_LENGTH_WEIGHT     = 0.20   # contribution of post length (proxy for distress)
SEVERITY_KEYWORD_WEIGHT    = 0.40   # contribution of strong negative keywords

# Maximum post word count used for length normalisation in severity scoring
SEVERITY_LENGTH_CAP = 500   # posts longer than this are capped at 1.0 on length axis

# Strong negative / crisis keywords for severity keyword scoring
# Presence of each keyword adds to the raw keyword score before normalisation
SEVERITY_KEYWORDS = [
    # hopelessness / suicidality
    "suicid", "hopeless", "worthless", "numb", "emptiness", "end life",
    "want die", "kill myself", "no point", "give up",
    # trauma / crisis
    "trauma", "abuse", "assault", "nightmare", "flashback", "trigger",
    "panic", "dissociat", "breakdown", "crisis",
    # severe depression / anxiety
    "cant breathe", "cant sleep", "cant eat", "self harm", "cutting",
    "isolat", "alone", "numb", "dread", "terror",
    # ADHD / OCD distress
    "spiral", "obsess", "intrusive", "compuls", "paralys", "shame",
]

# LDA topic modeling
LDA_N_TOPICS      = 7     # number of topics to extract per class
LDA_N_TOP_WORDS   = 10    # top words shown per topic
LDA_MAX_ITER      = 20
LDA_RANDOM_STATE  = 42

# Insights: top N terms to highlight per class in the insights report
INSIGHTS_TOP_N    = 8

# ─────────────────────────────────────────────
# API CONFIGURATION
# ─────────────────────────────────────────────

API_CONFIG = {
    "model_backend": "classical",
    "classical_model_name": "best_model",
    "host": "0.0.0.0",
    "port": 8000,
}

# ─────────────────────────────────────────────
# PATHS — unified dict for modules that expect it
# ─────────────────────────────────────────────

PATHS = {
    "root_dir": ROOT_DIR,
    "data_dir": DATA_DIR,
    "processed_dir": PROCESSED_DIR,
    "splits_dir": SPLITS_DIR,
    "models_dir": MODELS_DIR,
    "reports_dir": REPORTS_DIR,
    "figures_dir": FIGURES_DIR,
    "metrics_dir": METRICS_DIR,
    "features_dir": FEATURES_DIR,

    # 🔥 ADD THESE (CRITICAL — from Claude config)
    "merged_raw": MERGED_RAW_PATH,
    "cleaned": CLEANED_DATA_PATH,
    "train": TRAIN_PATH,
    "val": VAL_PATH,
    "test": TEST_PATH,
    "tfidf_vectorizer": FEATURE_VEC_PATH,
}