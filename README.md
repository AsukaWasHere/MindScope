# 🧠 MindScope: Reddit Mental Health NLP System

> A production-ready, end-to-end NLP pipeline for multi-class mental health condition classification — from raw Reddit data to a deployed FastAPI backend and Streamlit demo UI.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## ⚠️ Ethical Disclaimer

> This system is **NOT a clinical tool**. Predictions must never be used to diagnose, label, or make decisions about real individuals. All outputs are research artefacts derived from community language patterns — not medical assessments.

---

## Table of Contents

- [A. Project Overview](#a-project-overview)
- [B. What Was Built](#b-what-was-built)
- [C. Dataset](#c-dataset)
- [D. Project Structure](#d-project-structure)
- [E. Pipeline Design](#e-pipeline-design)
- [F. BERT Add-On](#f-bert-add-on)
- [G. FastAPI Backend](#g-fastapi-backend)
- [H. Streamlit Frontend](#h-streamlit-frontend)
- [I. Deployment](#i-deployment)
- [J. Modeling Objectives & Metrics](#j-modeling-objectives--metrics)
- [K. Quick Start](#k-quick-start)
- [L. Development Rules](#l-development-rules)

---

## A. Project Overview

### Problem Statement

Mental health conditions such as ADHD, Asperger's, Depression, OCD, and PTSD are often underdiagnosed or misunderstood. Millions of people turn to online communities to share their experiences, seek support, and describe their symptoms — often before ever speaking to a clinician.

MindScope builds a supervised NLP classification system that:
- Identifies which mental health condition a Reddit post most likely relates to
- Extracts linguistic and semantic signals that distinguish each condition
- Serves predictions via a REST API with sentiment analysis and severity estimation
- Provides a browser-based Streamlit demo for interactive exploration

### Why This Matters

| Motivation | Detail |
|---|---|
| **Early signal detection** | Language patterns may surface distress before clinical contact |
| **Research acceleration** | NLP-derived features enable large-scale studies without manual annotation |
| **Stigma reduction** | Understanding self-description builds empathetic communication tools |
| **Support triage** | Future applications could route users to relevant resources automatically |

### Real-World Use Cases

| Use Case | Description |
|---|---|
| Mental health chatbot routing | Classify user input to deliver condition-specific responses |
| Clinical research support | Identify linguistic biomarkers for each condition |
| Content moderation assistance | Flag high-distress posts in support communities |
| Public health monitoring | Track condition discourse trends over time |
| Psychoeducation tools | Generate condition-aware summaries for caregivers |

---

## B. What Was Built

This project was developed in four progressive phases:

### Phase 1 — NLP Pipeline (Core)
A 7-stage end-to-end pipeline: data ingestion → cleaning → text preprocessing → feature engineering → model training → evaluation → inference. Supports Logistic Regression, Random Forest, and LightGBM classifiers over TF-IDF features.

### Phase 2 — BERT Fine-Tuning (Add-On)
`bert-base-uncased` fine-tuned for 5-class mental health classification using HuggingFace Transformers. Plugs into the existing pipeline via a `--use_bert` flag — does not replace classical models.

### Phase 3 — FastAPI Backend
A production REST API wrapping the trained model. Returns predicted condition, confidence scores, VADER sentiment analysis, and a heuristic severity estimate. Supports both classical and BERT backends via a single config toggle.

### Phase 4 — Streamlit Frontend + Deployment
A Streamlit web UI that calls the FastAPI backend. Includes deployment infrastructure for Render (API + frontend) and HuggingFace Spaces (frontend), with lean per-environment requirements files and a one-shot setup script.

---

## C. Dataset

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
| `author` | string | Reddit username (anonymized by platform) |
| `body` | string | **Primary text feature** — full post content |
| `created_utc` | integer | Unix timestamp of post creation |
| `id` | string | Unique Reddit post identifier |
| `num_comments` | integer | Number of comments received |
| `score` | integer | Net upvotes (upvotes minus downvotes) |
| `subreddit` | string | Name of the subreddit — used as the **class label** |
| `title` | string | **Secondary text feature** — post headline |
| `upvote_ratio` | float | Proportion of votes that were upvotes (0.0–1.0) |
| `url` | string | Direct link to the Reddit post |

**Text Feature Strategy**: The primary model input is `title + " " + body` — a combined field that captures both headline intent and full narrative.

### Dataset Biases and Limitations

| Bias / Limitation | Explanation |
|---|---|
| **Self-selection bias** | Posters are not representative of all people with these conditions |
| **Diagnosis uncertainty** | Subreddit membership ≠ clinical diagnosis |
| **Comorbidity overlap** | ADHD + depression, OCD + PTSD etc. are common; posts may fit multiple classes |
| **Temporal drift** | Language norms shift over time |
| **Deleted content** | Posts with `[deleted]` or `[removed]` bodies are filtered |
| **Class imbalance** | Subreddits vary in size |
| **No ground truth** | Labels are subreddit-derived, not clinician-validated |
| **Author leakage** | Same author may appear in train and test sets |

---

## D. Project Structure

```
mindscope/
│
├── data/
│   ├── raw/                         # Original unmodified CSVs — never touched
│   │   ├── adhd.csv
│   │   ├── aspergers.csv
│   │   ├── depression.csv
│   │   ├── ocd.csv
│   │   └── ptsd.csv
│   ├── processed/
│   │   ├── merged_raw.csv           # All 5 CSVs merged with 'label' column
│   │   └── cleaned.csv              # After cleaning pipeline
│   └── splits/
│       ├── train.csv                # 70% stratified
│       ├── val.csv                  # 15% stratified
│       └── test.csv                 # 15% stratified
│
├── src/
│   ├── ingestion/
│   │   └── loader.py                # Load CSVs, assign labels, merge
│   ├── preprocessing/
│   │   ├── cleaner.py               # Null removal, deduplication, splitting
│   │   └── text_processor.py        # Tokenize, lowercase, lemmatize
│   ├── features/
│   │   ├── tfidf_vectorizer.py      # TF-IDF feature extraction
│   │   ├── embedder.py              # Sentence-transformer embeddings
│   │   └── bert_dataset.py          # ★ NEW — PyTorch Dataset for BERT
│   ├── modeling/
│   │   ├── trainer.py               # Classical model training
│   │   ├── evaluator.py             # Metrics + confusion matrix
│   │   ├── predictor.py             # Load model + run inference
│   │   ├── bert_trainer.py          # ★ NEW — BERT fine-tuning loop
│   │   └── bert_evaluator.py        # ★ NEW — BERT evaluation + metrics
│   ├── analysis/
│   │   ├── eda.py                   # EDA visualizations
│   │   └── topic_model.py           # LDA topic modelling per class
│   └── utils/
│       ├── config.py                # All constants, hyperparams, API_CONFIG, BERT_CONFIG
│       ├── logger.py                # Centralised logging (no bare print())
│       └── helpers.py               # save/load/ensure_dir/label_encoder
│
├── api/                             # ★ NEW — FastAPI backend
│   ├── __init__.py
│   ├── app.py                       # Route definitions
│   ├── inference.py                 # Preprocess → predict → sentiment → severity
│   ├── model_loader.py              # Load artifacts once at startup
│   └── schemas.py                   # Pydantic request/response models
│
├── models/
│   └── saved/                       # Serialized model artifacts (.pkl, BERT dir)
│       ├── lightgbm.pkl
│       ├── tfidf_vectorizer.pkl
│       └── bert/                    # ★ NEW — HuggingFace model + tokenizer
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
│
├── reports/
│   ├── figures/                     # Confusion matrices, word clouds, plots
│   └── metrics/                     # JSON metric files per model
│
├── tests/
│   ├── test_cleaner.py
│   ├── test_text_processor.py
│   ├── test_trainer.py
│   └── test_api.py                  # ★ NEW — FastAPI endpoint tests
│
├── app.py                           # ★ NEW — Root shim: uvicorn app:app --reload
├── streamlit_app.py                 # ★ NEW — Streamlit demo UI
├── run_pipeline.py                  # CLI entrypoint (supports --use_bert flag)
├── setup_env.sh                     # ★ NEW — One-shot environment setup script
├── render.yaml                      # ★ NEW — Render.com infrastructure-as-code
├── Procfile                         # ★ NEW — Render fallback start command
│
├── requirements.txt                 # Full training + dev dependencies
├── requirements_api.txt             # ★ NEW — Lean API-only deps (~400 MB)
├── requirements_streamlit.txt       # ★ NEW — Streamlit + viz deps
│
├── .gitignore
├── README.md                        # THIS FILE
└── SYSTEM_FLOW.md                   # Data and module flow documentation
```

> ★ NEW = added after the initial pipeline build

---

## E. Pipeline Design

### System Flow

```
data/raw/*.csv
     │
     ▼
[1] loader.py           ──► data/processed/merged_raw.csv
     │
     ▼
[2] cleaner.py          ──► data/processed/cleaned.csv
                        ──► data/splits/{train,val,test}.csv
     │
     ▼
[3] text_processor.py   ──► clean_text (in-memory)
     │
     ┌──────────────────────┐
     ▼                      ▼
[4a] tfidf_vectorizer   [4b] embedder.py / bert_dataset.py
     │                      │
     └───────────┬───────────┘
                 ▼
[5] trainer.py / bert_trainer.py   ──► models/saved/<model>.pkl  or  bert/
                 │
                 ▼
[6] evaluator.py / bert_evaluator.py ──► reports/metrics/<model>_metrics.json
                                     ──► reports/figures/<model>_confusion_matrix.png
                 │
                 ▼
[7] predictor.py  ──►  api/inference.py  ──►  POST /predict  ──►  streamlit_app.py
```

### Stage-by-Stage Summary

| Stage | Module | Input | Output |
|---|---|---|---|
| 1. Ingestion | `loader.py` | `data/raw/*.csv` | `merged_raw.csv` |
| 2. Cleaning | `cleaner.py` | `merged_raw.csv` | `cleaned.csv` + splits |
| 3. Text Preprocessing | `text_processor.py` | raw `text` string | `clean_text` string |
| 4a. TF-IDF | `tfidf_vectorizer.py` | `clean_text` | sparse matrix + `.pkl` |
| 4b. Embeddings | `embedder.py` | `clean_text` | dense numpy array |
| 5. Training | `trainer.py` | feature matrix + labels | model `.pkl` |
| 6. Evaluation | `evaluator.py` | predictions + labels | metrics JSON + PNG |
| 7. Inference | `predictor.py` | raw text | class + confidence scores |

### Text Preprocessing Steps (`text_processor.preprocess()`)

1. Lowercase all text
2. Remove URLs (`http://...`, `www...`)
3. Remove HTML tags (`<br>`, `&amp;`, etc.)
4. Remove Reddit artifacts (`/r/`, `/u/`, subreddit mentions)
5. Remove punctuation and special characters
6. Tokenize into words
7. Remove English stopwords (NLTK)
8. Lemmatize tokens (spaCy / NLTK WordNetLemmatizer)
9. Rejoin into clean string

> This function is **pure and stateless** — same input always produces the same output.

### TF-IDF Parameters

```python
TfidfVectorizer(
    max_features = 50_000,
    ngram_range  = (1, 2),
    sublinear_tf = True,
)
```

### Classical Models

| Priority | Model | Feature Type | Saved As |
|---|---|---|---|
| 1 | Logistic Regression | TF-IDF | `logistic_regression.pkl` |
| 2 | Random Forest | TF-IDF | `random_forest.pkl` |
| 3 | LightGBM | TF-IDF | `lightgbm.pkl` |
| 4 | BERT (fine-tuned) | Raw text | `models/saved/bert/` |

---

## F. BERT Add-On

`bert-base-uncased` fine-tuned for 5-class classification using HuggingFace Transformers.

### New Modules

| File | Purpose |
|---|---|
| `src/features/bert_dataset.py` | PyTorch `Dataset` — tokenizes, pads, truncates sequences |
| `src/modeling/bert_trainer.py` | Fine-tuning loop with AdamW + linear warmup scheduler |
| `src/modeling/bert_evaluator.py` | Inference on test set + saves metrics JSON + confusion matrix |

### BERT Configuration (`src/utils/config.py`)

```python
BERT_CONFIG = {
    "model_name":        "bert-base-uncased",
    "save_dir":          "models/saved/bert",
    "max_length":        256,     # tokens per sequence
    "epochs":            3,       # 2–4 is sufficient for fine-tuning
    "batch_size":        16,
    "learning_rate":     2e-5,
    "weight_decay":      0.01,
    "warmup_ratio":      0.1,
}
```

### Running BERT

```bash
# Train classical + BERT
python run_pipeline.py --model lightgbm --features tfidf --use_bert

# BERT only
python run_pipeline.py --use_bert --skip_classical
```

### Architecture

```
bert-base-uncased
      │
  [CLS] token hidden state (768-dim)
      │
  Linear layer (768 → 5)
      │
  CrossEntropyLoss
```

---

## G. FastAPI Backend

### Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Welcome message |
| `GET` | `/health` | Liveness check + loaded model info |
| `GET` | `/classes` | List of 5 supported condition labels |
| `POST` | `/predict` | Classify text + sentiment + severity |
| `GET` | `/docs` | Swagger UI (auto-generated) |

### `/predict` Request & Response

**Request:**
```json
{
  "text": "I feel anxious all the time and can't stop worrying."
}
```

**Response:**
```json
{
  "prediction":       "ocd",
  "confidence":       0.7241,
  "all_scores":       { "adhd": 0.04, "aspergers": 0.03, "depression": 0.12, "ocd": 0.72, "ptsd": 0.09 },
  "sentiment":        "negative",
  "sentiment_scores": { "neg": 0.38, "neu": 0.62, "pos": 0.0, "compound": -0.51 },
  "severity":         "moderate",
  "clean_text":       "feel anxious time stop worry"
}
```

### Inference Pipeline (per request)

```
raw text
   │
   ▼
text_processor.preprocess()     ← reuses existing module
   │
   ▼
tfidf_vectorizer.transform()    ← classical path
 OR bert tokenizer + forward()  ← BERT path
   │
   ▼
model.predict_proba()
   │
   ▼
VADER sentiment analysis        ← run on original raw text
   │
   ▼
severity heuristic              ← compound score × confidence × class weight
   │
   ▼
JSON response
```

### Backend Toggle

Switch between classical and BERT without changing any route code:

```python
# src/utils/config.py
API_CONFIG = {
    "model_backend":        "classical",  # or "bert"
    "classical_model_name": "lightgbm",
}
```

### Module Responsibilities

| Module | Does | Does NOT do |
|---|---|---|
| `api/app.py` | Define routes | Any ML logic |
| `api/inference.py` | Run full inference pipeline | Load models |
| `api/model_loader.py` | Load artifacts at startup | Run inference |
| `api/schemas.py` | Define request/response shapes | Any logic |

---

## H. Streamlit Frontend

`streamlit_app.py` provides an interactive web UI that:
- Accepts freeform text input (with example presets)
- Calls `POST /predict` on the FastAPI backend
- Displays: predicted condition, confidence bar chart (Plotly), sentiment breakdown, severity badge
- Shows preprocessed text and raw JSON for transparency

The backend URL is configured via the `FASTAPI_URL` environment variable:

```bash
# Local development (default)
FASTAPI_URL=http://localhost:8000

# Production (set in Render / HF Spaces environment)
FASTAPI_URL=https://mindscope-api.onrender.com
```

---

## I. Deployment

### Requirements Strategy

Three separate requirements files — use the right one for each context:

| File | Use When | Includes PyTorch? | Approx Size |
|---|---|---|---|
| `requirements.txt` | Local dev + training | Yes | ~3 GB |
| `requirements_api.txt` | FastAPI server on Render | Optional | ~400 MB |
| `requirements_streamlit.txt` | Streamlit / HF Spaces | No | ~300 MB |

### FastAPI → Render

```bash
# Render Build Command
pip install -r requirements_api.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm
wget -O models/saved/lightgbm.pkl <YOUR_RELEASE_URL>
wget -O models/saved/tfidf_vectorizer.pkl <YOUR_RELEASE_URL>

# Render Start Command
uvicorn app:app --host 0.0.0.0 --port $PORT
```

### Streamlit → HuggingFace Spaces

1. Create a new Space (SDK: Streamlit)
2. Connect your GitHub repo
3. Set secret: `FASTAPI_URL=https://mindscope-api.onrender.com`
4. HF Spaces auto-deploys on every push to `main`

### Both Services → Render via Blueprint

```bash
# From project root — Render reads render.yaml automatically
git push origin main
# Then: Render Dashboard → Blueprints → New Blueprint Instance → select repo
```

### Local Setup (one command)

```bash
chmod +x setup_env.sh && ./setup_env.sh
```

The script creates a venv, installs all dependencies, downloads NLTK corpora and the spaCy model, and creates a `.env` file.

---

## J. Modeling Objectives & Metrics

### Primary Task: 5-Class Classification

- **Input**: Reddit post text (title + body combined)
- **Output**: One of `adhd | aspergers | depression | ocd | ptsd`
- **Baseline**: Majority class classifier (~20% accuracy on balanced data)
- **Target**: Macro F1 ≥ 0.75

### Metrics

| Metric | Purpose |
|---|---|
| **Accuracy** | Overall correctness — misleading with class imbalance alone |
| **Precision (per class)** | Of posts predicted as X, how many truly are X? |
| **Recall (per class)** | Of all true X posts, how many were correctly caught? |
| **F1 (per class)** | Harmonic mean of precision and recall |
| **Macro F1** | Average F1 across all 5 classes — primary evaluation metric |
| **Confusion Matrix** | Visual map of which classes are confused with each other |

### Output Files

| Output | Location |
|---|---|
| Metrics JSON | `reports/metrics/<model_name>_metrics.json` |
| Confusion matrix PNG | `reports/figures/<model_name>_confusion_matrix.png` |
| Training logs | stdout via `src/utils/logger.py` |

### Additional Analysis

- **Sentiment analysis**: VADER polarity distribution per class (analytical only — not a training label)
- **Topic modelling**: LDA per class via `src/analysis/topic_model.py` — outputs top N words per topic per class

---

## K. Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/mindscope.git
cd mindscope
chmod +x setup_env.sh && ./setup_env.sh
source venv/bin/activate
```

### 2. Add your data

```bash
# Place the 5 CSV files in:
data/raw/adhd.csv
data/raw/aspergers.csv
data/raw/depression.csv
data/raw/ocd.csv
data/raw/ptsd.csv
```

### 3. Run the full pipeline

```bash
# Classical ML (LightGBM + TF-IDF)
python run_pipeline.py --model lightgbm --features tfidf

# With BERT fine-tuning
python run_pipeline.py --model lightgbm --features tfidf --use_bert

# All classical models
python run_pipeline.py --model all --features tfidf
```

### 4. Start the API

```bash
uvicorn app:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs
```

### 5. Start the Streamlit UI

```bash
# In a second terminal
streamlit run streamlit_app.py
# → http://localhost:8501
```

### 6. Test a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel hopeless and cannot get out of bed anymore."}'
```

---

## L. Development Rules

These rules are **non-negotiable** and apply to all code in `src/` and `api/`:

### 1. No Duplicate Logic
Every function exists in exactly one place. Shared helpers go in `src/utils/helpers.py`.

### 2. Reusable, Parameterised Modules
No hardcoded values inline. All constants live in `src/utils/config.py`.

### 3. Strict Separation of Concerns

| Module | Only Does |
|---|---|
| `loader.py` | Loads data |
| `cleaner.py` | Cleans data |
| `text_processor.py` | Preprocesses text |
| `tfidf_vectorizer.py` / `bert_dataset.py` | Feature engineering |
| `trainer.py` / `bert_trainer.py` | Training only |
| `evaluator.py` / `bert_evaluator.py` | Evaluation only |
| `api/app.py` | Route definitions only |
| `api/inference.py` | Inference logic only |
| `api/model_loader.py` | Artifact loading only |

### 4. Follow the Defined Architecture
No new folders or files without justification. Notebooks are for exploration — no production logic lives there.

### 5. Reproducibility
All random seeds set via `config.py`. Train/val/test splits are saved to disk and reused — never regenerated per run.

### 6. Logging Over Printing
All `src/` and `api/` modules use `src/utils/logger.py`. No bare `print()` statements.

### 7. Data Immutability
`data/raw/` is never modified. All transformations produce new files in `data/processed/` or `data/splits/`.

### 8. No Circular Imports
`utils/` modules never import from `src/` subfolders. All dependency flow is top-down.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data | pandas, numpy |
| NLP preprocessing | NLTK, spaCy |
| Feature engineering | scikit-learn (TF-IDF), sentence-transformers |
| Classical ML | scikit-learn, LightGBM, XGBoost |
| Deep learning | PyTorch, HuggingFace Transformers (bert-base-uncased) |
| Sentiment analysis | VADER (vaderSentiment) |
| Topic modelling | Gensim (LDA) |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit, Plotly |
| Deployment | Render, HuggingFace Spaces |
| Testing | pytest, FastAPI TestClient |
| Logging | Python `logging` module |

---

*MindScope — Research NLP system. Not a clinical tool.*