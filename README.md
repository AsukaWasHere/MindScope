# 🧠 MindScope: Reddit Mental Health NLP System

> A production-ready NLP pipeline for multi-class mental health condition classification using Reddit community data.

---

## A. Project Overview

### Problem Statement

Mental health conditions such as ADHD, Asperger's, Depression, OCD, and PTSD are often underdiagnosed or misunderstood. Millions of people turn to online communities to share their experiences, seek support, and describe their symptoms — often before ever speaking to a clinician.

This project builds a supervised NLP classification system that:
- Identifies which mental health condition a Reddit post most likely relates to
- Extracts linguistic and semantic signals that distinguish each condition
- Provides interpretable insights into the language patterns of each community

### Why Mental Health NLP Matters

- **Early signal detection**: Language patterns in posts may surface distress before clinical contact
- **Research acceleration**: NLP-derived features enable large-scale behavioral studies without manual annotation
- **Stigma reduction**: Understanding how communities self-describe helps build empathetic, accurate communication tools
- **Support triage**: Future applications could help route users to relevant resources automatically

### Real-World Use Cases

| Use Case | Description |
|---|---|
| Mental health chatbot routing | Classify user input to deliver condition-specific responses |
| Clinical research support | Identify linguistic biomarkers for each condition |
| Content moderation assistance | Flag high-distress posts in support communities |
| Public health monitoring | Track condition discourse trends over time |
| Psychoeducation tools | Generate condition-aware summaries for caregivers |

---

## B. Dataset Explanation

### Source
Reddit posts scraped from five mental health subreddits. Each row is a single post.

### Files
| File | Subreddit Class | Description |
|---|---|---|
| `adhd.csv` | ADHD | Posts from r/ADHD — focus, impulsivity, executive dysfunction |
| `aspergers.csv` | Aspergers | Posts from r/aspergers — social difficulty, sensory issues, ASD identity |
| `depression.csv` | Depression | Posts from r/depression — low mood, hopelessness, anhedonia |
| `ocd.csv` | OCD | Posts from r/OCD — intrusive thoughts, compulsions, rituals |
| `ptsd.csv` | PTSD | Posts from r/PTSD — trauma, flashbacks, hypervigilance |

### Column Definitions

| Column | Type | Description |
|---|---|---|
| `author` | string | Reddit username of the poster (anonymized by platform) |
| `body` | string | **Primary text feature** — the full content of the post |
| `created_utc` | integer | Unix timestamp when the post was created |
| `id` | string | Unique Reddit post identifier |
| `num_comments` | integer | Number of comments the post received |
| `score` | integer | Net upvotes (upvotes minus downvotes) |
| `subreddit` | string | Name of the subreddit (used as the **class label**) |
| `title` | string | **Secondary text feature** — the post headline |
| `upvote_ratio` | float | Proportion of votes that were upvotes (0.0–1.0) |
| `url` | string | Direct link to the Reddit post |

### Text Feature Strategy
The primary modeling input will be a **combined text field**: `title + " " + body`. This captures both the headline intent and the detailed narrative.

### Dataset Biases and Limitations

| Bias / Limitation | Explanation |
|---|---|
| **Self-selection bias** | Users who post are not representative of all people with these conditions |
| **Diagnosis uncertainty** | Subreddit membership ≠ clinical diagnosis. A user in r/depression may have PTSD |
| **Comorbidity overlap** | ADHD + depression, OCD + PTSD etc. are common; posts may fit multiple classes |
| **Temporal drift** | Language norms and community culture shift over time |
| **Deleted content** | Posts with `[deleted]` or `[removed]` bodies must be filtered |
| **Class imbalance** | Subreddits vary in size; some classes may be underrepresented |
| **No ground truth** | Labels are subreddit-derived, not clinician-validated |
| **Author leakage** | Same author may appear in train and test; shuffle carefully |

> ⚠️ **Ethical note**: This system is NOT a clinical tool. Predictions must never be used to diagnose, label, or make decisions about real individuals.

---

## C. Final Project Structure

```
mindscope/
│
├── data/
│   ├── raw/                        # Original unmodified CSV files
│   │   ├── adhd.csv
│   │   ├── aspergers.csv
│   │   ├── depression.csv
│   │   ├── ocd.csv
│   │   └── ptsd.csv
│   ├── processed/                  # Cleaned and merged data
│   │   ├── merged_raw.csv          # All files combined with 'label' column
│   │   └── cleaned.csv             # After cleaning pipeline
│   └── splits/                     # Train/val/test splits
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── src/
│   ├── ingestion/
│   │   └── loader.py               # Load and merge all CSVs; assign labels
│   │
│   ├── preprocessing/
│   │   ├── cleaner.py              # Remove nulls, duplicates, deleted posts
│   │   └── text_processor.py      # Tokenize, lowercase, remove noise, lemmatize
│   │
│   ├── features/
│   │   ├── tfidf_vectorizer.py     # TF-IDF feature extraction
│   │   └── embedder.py             # Sentence/transformer embeddings
│   │
│   ├── modeling/
│   │   ├── trainer.py              # Model training logic (all model types)
│   │   ├── evaluator.py            # Metrics: accuracy, F1, confusion matrix
│   │   └── predictor.py            # Load saved model and run inference
│   │
│   ├── analysis/
│   │   ├── eda.py                  # Exploratory data analysis + visualizations
│   │   └── topic_model.py          # LDA topic modeling per class
│   │
│   └── utils/
│       ├── config.py               # All global constants and hyperparameters
│       ├── logger.py               # Centralized logging setup
│       └── helpers.py              # Shared utility functions (e.g., save/load)
│
├── models/
│   └── saved/                      # Serialized trained models (.pkl, .pt, etc.)
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory analysis with plots
│   ├── 02_preprocessing.ipynb      # Cleaning and preprocessing walkthrough
│   ├── 03_modeling.ipynb           # Model training experiments
│   └── 04_evaluation.ipynb         # Evaluation and error analysis
│
├── reports/
│   ├── figures/                    # Saved plots (confusion matrix, word clouds, etc.)
│   └── metrics/                    # JSON/CSV files with evaluation results
│
├── tests/
│   ├── test_cleaner.py
│   ├── test_text_processor.py
│   └── test_trainer.py
│
├── README.md                       # THIS FILE — master context document
├── SYSTEM_FLOW.md                  # Data and module flow documentation
├── requirements.txt                # All Python dependencies
└── run_pipeline.py                 # Single entrypoint to run the full pipeline
```

### Folder Purpose Summary

| Folder/File | Purpose |
|---|---|
| `data/raw/` | Immutable source files — never modified |
| `data/processed/` | Outputs from ingestion and cleaning steps |
| `data/splits/` | Fixed train/val/test sets for reproducibility |
| `src/ingestion/` | Data loading and label assignment |
| `src/preprocessing/` | Cleaning and NLP preprocessing |
| `src/features/` | Feature engineering (TF-IDF, embeddings) |
| `src/modeling/` | Training, evaluation, and inference |
| `src/analysis/` | EDA and unsupervised topic analysis |
| `src/utils/` | Shared config, logging, helpers |
| `models/saved/` | Persisted model artifacts |
| `notebooks/` | Exploratory and experimental work |
| `reports/` | Saved outputs for documentation/sharing |
| `tests/` | Unit tests for core modules |
| `run_pipeline.py` | CLI entrypoint to run full end-to-end pipeline |

---

## D. Full Pipeline Design

### Step 1 — Data Ingestion (`src/ingestion/loader.py`)
- Load each CSV file using `pandas`
- Add a `label` column with the subreddit name (e.g., `"depression"`)
- Concatenate all five DataFrames into one: `merged_raw.csv`
- Log class distribution

**Input**: `data/raw/*.csv`
**Output**: `data/processed/merged_raw.csv`

---

### Step 2 — Data Cleaning (`src/preprocessing/cleaner.py`)
- Drop rows where `body` is null, `[deleted]`, or `[removed]`
- Remove duplicate posts by `id`
- Drop rows with empty `title`
- Filter out posts with extremely short body text (< 10 words)
- Combine `title` and `body` into a single `text` column
- Save to `data/processed/cleaned.csv`
- Generate train/val/test splits (70/15/15) stratified by label → `data/splits/`

**Input**: `data/processed/merged_raw.csv`
**Output**: `data/processed/cleaned.csv`, `data/splits/*.csv`

---

### Step 3 — Text Preprocessing (`src/preprocessing/text_processor.py`)
Applied to the combined `text` column:
- Lowercase all text
- Remove URLs, HTML tags, special characters
- Remove Reddit-specific artifacts (e.g., `/r/`, `/u/`)
- Tokenize using `nltk` or `spacy`
- Remove stopwords
- Lemmatize tokens
- Return clean string per post

**Input**: Raw `text` column (string)
**Output**: Preprocessed `clean_text` column (string)

---

### Step 4 — Feature Engineering (`src/features/`)

Two parallel strategies:

**4a. TF-IDF** (`tfidf_vectorizer.py`)
- Fit on training data only
- Transform train/val/test separately
- Parameters: `max_features=50000`, `ngram_range=(1,2)`, `sublinear_tf=True`

**4b. Transformer Embeddings** (`embedder.py`)
- Use `sentence-transformers` (e.g., `all-MiniLM-L6-v2`)
- Encode each post into a dense vector
- Used for deep learning or similarity-based classification

**Input**: `clean_text` column
**Output**: Feature matrix (sparse TF-IDF or dense embeddings)

---

### Step 5 — Modeling (`src/modeling/trainer.py`)

Models trained in priority order:

| Priority | Model | Feature Type |
|---|---|---|
| 1 | Logistic Regression | TF-IDF |
| 2 | Random Forest | TF-IDF |
| 3 | LightGBM / XGBoost | TF-IDF |
| 4 | Fine-tuned BERT / DistilBERT | Raw text |

- All models follow a unified training interface in `trainer.py`
- Hyperparameters stored in `src/utils/config.py`
- Models saved to `models/saved/`

---

### Step 6 — Evaluation (`src/modeling/evaluator.py`)
- Compute accuracy, per-class precision/recall/F1, macro F1
- Generate and save confusion matrix
- Save full classification report to `reports/metrics/`
- Compare multiple model results

**Input**: Predictions + true labels
**Output**: Metrics JSON, confusion matrix PNG

---

### Step 7 — Inference (`src/modeling/predictor.py`)
- Load a saved model and vectorizer/tokenizer
- Accept raw text input
- Return predicted class + confidence scores

---

### Optional: Deployment
- Wrap `predictor.py` in a FastAPI endpoint
- Containerize with Docker
- Serve via `/predict` POST endpoint accepting JSON `{"text": "..."}`

---

## E. Modeling Objectives

### Primary: Multi-Class Classification
- **Task**: Given a post's text, predict which of 5 subreddits it belongs to
- **Classes**: `adhd`, `aspergers`, `depression`, `ocd`, `ptsd`
- **Baseline**: Majority class classifier
- **Target**: Macro F1 ≥ 0.75

### Optional: Binary Classification
- **Examples**:
  - Depression vs. Not-Depression (one-vs-rest)
  - Internalizing (depression, ptsd) vs. Externalizing (adhd, ocd)
- **Purpose**: Simpler models for specific triage use cases

### Sentiment Analysis
- Use `VADER` or `TextBlob` on the `body` field
- Analyze sentiment polarity distribution per class
- Not a training label — used as an analytical feature

### Topic Modeling
- Apply `LDA` (Latent Dirichlet Allocation) per class
- Discover recurring themes within each subreddit
- Implemented in `src/analysis/topic_model.py`
- Outputs top N words per topic per class

---

## F. Metrics

### Classification Metrics (Primary)

| Metric | Why It Matters |
|---|---|
| **Accuracy** | Overall correctness; useful baseline but misleading with class imbalance |
| **Precision (per class)** | Of all posts predicted as class X, how many truly are X? |
| **Recall (per class)** | Of all true class X posts, how many did we correctly catch? |
| **F1 Score (per class)** | Harmonic mean of precision and recall — balanced measure |
| **Macro F1** | Average F1 across all classes equally — handles imbalance |
| **Confusion Matrix** | Shows which classes are confused with each other |

### Reporting
- All metrics saved to `reports/metrics/` as JSON
- Confusion matrix saved as PNG to `reports/figures/`
- Training logs written via `src/utils/logger.py`

---

## G. Development Rules

These rules are **non-negotiable** and must be followed in all future code:

### 1. No Duplicate Logic
- Every function exists in exactly one place
- If two modules need the same helper, it goes in `src/utils/helpers.py`

### 2. Reusable Modules Only
- Functions must be parameterized, not hardcoded
- No inline magic values — all constants live in `src/utils/config.py`

### 3. Clear Separation of Concerns
- `loader.py` only loads data — it does not clean
- `cleaner.py` only cleans — it does not preprocess text
- `text_processor.py` only processes text — it does not vectorize
- `trainer.py` trains — it does not evaluate
- `evaluator.py` evaluates — it does not train

### 4. Follow the Defined Architecture Strictly
- No new folders or files without justification
- All imports use the defined module paths
- Notebooks are for exploration only — no production logic lives there

### 5. Reproducibility
- All random seeds set via `config.py`
- Train/val/test splits are saved to disk and reused — never regenerated per run

### 6. Logging Over Printing
- Use `src/utils/logger.py` for all output — no bare `print()` statements in `src/`

### 7. Data Immutability
- Files in `data/raw/` are never modified or overwritten
- All transformations produce new files in `data/processed/` or `data/splits/`