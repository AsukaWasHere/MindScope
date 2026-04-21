# SYSTEM_FLOW.md — MindScope Data & Module Flow

> This document describes exactly how data moves through the MindScope pipeline, which module calls which, and what each stage receives and produces.

---

## High-Level Overview

```
data/raw/*.csv
     │
     ▼
[1] loader.py          ──► data/processed/merged_raw.csv
     │
     ▼
[2] cleaner.py         ──► data/processed/cleaned.csv
                       ──► data/splits/train.csv, val.csv, test.csv
     │
     ▼
[3] text_processor.py  ──► clean_text column (in-memory / applied to splits)
     │
     ┌──────────────────┐
     ▼                  ▼
[4a] tfidf_vectorizer  [4b] embedder.py
     │                  │
     └────────┬─────────┘
              ▼
[5] trainer.py         ──► models/saved/<model_name>.pkl
              │
              ▼
[6] evaluator.py       ──► reports/metrics/<model_name>_metrics.json
                       ──► reports/figures/confusion_matrix.png
              │
              ▼
[7] predictor.py       ──► Predicted label + confidence (live inference)
```

---

## Stage-by-Stage Breakdown

---

### Stage 1 — Data Ingestion
**Module**: `src/ingestion/loader.py`

**Triggered by**: `run_pipeline.py` or manually

**What it does**:
1. Reads each of the 5 CSV files from `data/raw/`
2. Adds a `label` column to each DataFrame using the filename (e.g., `adhd.csv` → label = `"adhd"`)
3. Concatenates all 5 DataFrames into one unified DataFrame
4. Saves the result to `data/processed/merged_raw.csv`
5. Logs the total row count and per-class distribution

**Input**:
```
data/raw/adhd.csv
data/raw/aspergers.csv
data/raw/depression.csv
data/raw/ocd.csv
data/raw/ptsd.csv
```

**Output**:
```
data/processed/merged_raw.csv
Columns: author, body, created_utc, id, num_comments, score,
         subreddit, title, upvote_ratio, url, label
```

**Calls**: `src/utils/logger.py`, `src/utils/config.py`

---

### Stage 2 — Data Cleaning
**Module**: `src/preprocessing/cleaner.py`

**Triggered by**: `run_pipeline.py` after Stage 1

**What it does**:
1. Loads `data/processed/merged_raw.csv`
2. Removes rows where `body` is null, `"[deleted]"`, or `"[removed]"`
3. Removes rows where `title` is null or empty
4. Drops duplicate rows by `id`
5. Filters posts with fewer than 10 words in `body`
6. Creates a new column: `text = title + " " + body`
7. Saves cleaned data to `data/processed/cleaned.csv`
8. Performs stratified train/val/test split (70/15/15) by `label`
9. Saves splits to `data/splits/train.csv`, `val.csv`, `test.csv`

**Input**:
```
data/processed/merged_raw.csv
```

**Output**:
```
data/processed/cleaned.csv       ← Full cleaned dataset
data/splits/train.csv            ← 70% stratified
data/splits/val.csv              ← 15% stratified
data/splits/test.csv             ← 15% stratified
```

**Calls**: `src/utils/logger.py`, `src/utils/config.py` (for split ratios, random seed)

---

### Stage 3 — Text Preprocessing
**Module**: `src/preprocessing/text_processor.py`

**Triggered by**: Called by `trainer.py` and `predictor.py` as a preprocessing step on the `text` column

**What it does**:
Exposes a single function: `preprocess(text: str) -> str`

Steps inside `preprocess()`:
1. Lowercase the input string
2. Remove URLs (`http://...`, `www...`)
3. Remove HTML tags (`<br>`, `&amp;`, etc.)
4. Remove Reddit artifacts (`/r/`, `/u/`, subreddit mentions)
5. Remove punctuation and special characters
6. Tokenize into words
7. Remove English stopwords (via NLTK)
8. Lemmatize each token (via spaCy or NLTK WordNetLemmatizer)
9. Rejoin tokens into a clean string

**Input**:
```python
text: str  # Combined title + body string
```

**Output**:
```python
clean_text: str  # Normalized, tokenized, lemmatized string
```

**Calls**: No other project modules. Uses only NLTK / spaCy.

> ⚠️ This function is stateless and pure — same input always produces same output.

---

### Stage 4a — TF-IDF Feature Extraction
**Module**: `src/features/tfidf_vectorizer.py`

**Triggered by**: `trainer.py` when using classical ML models

**What it does**:
1. Receives `train`, `val`, `test` DataFrames (already cleaned)
2. Applies `text_processor.preprocess()` to the `text` column
3. Fits a `TfidfVectorizer` on training data only
4. Transforms all three splits
5. Saves the fitted vectorizer to `models/saved/tfidf_vectorizer.pkl`

**Input**:
```
train['text'], val['text'], test['text']  # Raw text strings
```

**Output**:
```
X_train, X_val, X_test   # Sparse matrices (scipy.sparse)
tfidf_vectorizer.pkl      # Saved to models/saved/
```

**Calls**: `src/preprocessing/text_processor.py`, `src/utils/config.py`, `src/utils/helpers.py`

---

### Stage 4b — Transformer Embeddings
**Module**: `src/features/embedder.py`

**Triggered by**: `trainer.py` when using embedding-based models

**What it does**:
1. Loads a pretrained `sentence-transformers` model (e.g., `all-MiniLM-L6-v2`)
2. Encodes each post's `text` into a dense vector
3. Returns numpy arrays for train/val/test

**Input**:
```
train['text'], val['text'], test['text']  # Raw text strings
```

**Output**:
```
X_train, X_val, X_test   # Dense numpy arrays (shape: N × embedding_dim)
```

**Calls**: `src/utils/config.py` (model name), `src/utils/logger.py`

> ℹ️ Stages 4a and 4b are **alternatives**, not sequential. `trainer.py` selects one based on configuration.

---

### Stage 5 — Model Training
**Module**: `src/modeling/trainer.py`

**Triggered by**: `run_pipeline.py`

**What it does**:
1. Reads config to determine which model(s) to train
2. Loads feature matrices from Stage 4 (TF-IDF or embeddings)
3. Instantiates model (LogReg, RandomForest, LightGBM, or DistilBERT)
4. Trains on `X_train`, `y_train`
5. Evaluates on `X_val` during training for early stopping (where applicable)
6. Saves trained model to `models/saved/<model_name>.pkl` or `.pt`

**Input**:
```
X_train, y_train   # Feature matrix + labels
X_val, y_val       # For validation during training
```

**Output**:
```
models/saved/logistic_regression.pkl
models/saved/random_forest.pkl
models/saved/lightgbm.pkl
models/saved/distilbert/             ← directory for HuggingFace models
```

**Calls**: `src/features/tfidf_vectorizer.py` or `src/features/embedder.py`, `src/utils/config.py`, `src/utils/logger.py`, `src/utils/helpers.py`

---

### Stage 6 — Evaluation
**Module**: `src/modeling/evaluator.py`

**Triggered by**: `run_pipeline.py` after training, or independently

**What it does**:
1. Loads a saved model from `models/saved/`
2. Runs predictions on `X_test`
3. Computes:
   - Accuracy
   - Precision, Recall, F1 per class
   - Macro-averaged F1
   - Confusion matrix
4. Saves metrics as JSON to `reports/metrics/`
5. Saves confusion matrix plot to `reports/figures/`
6. Logs all results via `logger.py`

**Input**:
```
X_test, y_test        # Test feature matrix + true labels
model_name: str       # Which saved model to evaluate
```

**Output**:
```
reports/metrics/<model_name>_metrics.json
reports/figures/<model_name>_confusion_matrix.png
```

**Calls**: `src/utils/logger.py`, `src/utils/helpers.py`

---

### Stage 7 — Inference
**Module**: `src/modeling/predictor.py`

**Triggered by**: External API call, script, or notebook

**What it does**:
1. Loads specified model from `models/saved/`
2. Loads corresponding vectorizer/tokenizer
3. Accepts raw text input
4. Runs `text_processor.preprocess()` on input
5. Vectorizes/embeds preprocessed text
6. Returns predicted class label + probability scores for all classes

**Input**:
```python
raw_text: str   # Any post text, no preprocessing required
```

**Output**:
```python
{
  "predicted_class": "depression",
  "confidence": 0.87,
  "all_scores": {
    "adhd": 0.04,
    "aspergers": 0.03,
    "depression": 0.87,
    "ocd": 0.02,
    "ptsd": 0.04
  }
}
```

**Calls**: `src/preprocessing/text_processor.py`, `src/features/tfidf_vectorizer.py` or `src/features/embedder.py`, `src/utils/helpers.py`, `src/utils/config.py`

---

## Utility Module Roles

### `src/utils/config.py`
- Central store for all constants and hyperparameters
- Called by every module — never imports from other `src/` modules
- Contains: file paths, model names, split ratios, random seeds, vectorizer params, embedding model names

### `src/utils/logger.py`
- Sets up a Python `logging` instance used across all modules
- All modules call `logger.info()`, `logger.warning()`, `logger.error()` — never `print()`
- Log format: `[TIMESTAMP] [MODULE] [LEVEL] message`

### `src/utils/helpers.py`
- Shared utility functions only:
  - `save_object(obj, path)` — pickle serialization
  - `load_object(path)` — pickle deserialization
  - `ensure_dir(path)` — create directory if not exists
  - `get_label_encoder()` — consistent label encoding across stages

---

## Analysis Modules (Non-Pipeline)

### `src/analysis/eda.py`
- Standalone: run manually from notebook or CLI
- Generates: class distribution bar charts, text length histograms, word clouds per class, score/engagement distributions
- Saves figures to `reports/figures/`

### `src/analysis/topic_model.py`
- Standalone: run after cleaning, before or after modeling
- Fits LDA model per class on cleaned text
- Outputs top N words per topic per class
- Saves results to `reports/metrics/topic_model_results.json`

---

## Pipeline Entry Point

### `run_pipeline.py`
Single script to execute the full pipeline end-to-end:

```
python run_pipeline.py --model lightgbm --features tfidf
```

**Execution order**:
```
loader.py
  → cleaner.py
    → text_processor.py (applied inside feature step)
      → tfidf_vectorizer.py or embedder.py
        → trainer.py
          → evaluator.py
```

Each stage is only re-run if its output files don't already exist (cache-friendly).

---

## Module Dependency Map

```
config.py ◄──── (all modules import from here)
logger.py ◄──── (all modules import from here)
helpers.py ◄─── (loader, cleaner, trainer, evaluator, predictor)

loader.py
  └── uses: config, logger

cleaner.py
  └── uses: config, logger, helpers

text_processor.py
  └── uses: (external: nltk/spacy only)

tfidf_vectorizer.py
  └── uses: text_processor, config, helpers, logger

embedder.py
  └── uses: config, logger

trainer.py
  └── uses: tfidf_vectorizer OR embedder, config, logger, helpers

evaluator.py
  └── uses: config, logger, helpers

predictor.py
  └── uses: text_processor, tfidf_vectorizer OR embedder, helpers, config
```

> **Rule**: No circular imports. `utils/` modules never import from `src/` subfolders.