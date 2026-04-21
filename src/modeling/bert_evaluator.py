"""
src/modeling/bert_evaluator.py
───────────────────────────────
Evaluation logic specific to the fine-tuned BERT model.

Responsibilities (matches evaluator.py separation of concerns)
──────────────────────────────────────────────────────────────
  - Load saved BERT model + tokenizer from disk
  - Run inference on the test split
  - Compute accuracy, per-class F1, macro F1, confusion matrix
  - Save metrics JSON + confusion matrix PNG (same format as evaluator.py)

This module does NOT train. It only evaluates.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.features.bert_dataset import MentalHealthBERTDataset
from src.utils.config import BERT_CONFIG, PATHS
from src.utils import config
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
def evaluate_bert(test_df, label_encoder):
    """
    Load saved BERT model and evaluate on the test set.

    Parameters
    ----------
    test_df       : pd.DataFrame with 'clean_text' and 'label' columns
    label_encoder : fitted sklearn LabelEncoder

    Returns
    -------
    dict — metrics including accuracy and macro_f1
    """
    cfg    = BERT_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[bert_evaluator] Using device: {device}")

    # ── Load model + tokenizer ────────────────────────────────────────
    save_dir = cfg["save_dir"]
    logger.info(f"[bert_evaluator] Loading model from: {save_dir}")
    tokenizer = BertTokenizer.from_pretrained(save_dir)
    model     = BertForSequenceClassification.from_pretrained(save_dir)
    model.to(device)
    model.eval()

    # ── Build test DataLoader ─────────────────────────────────────────
    y_test = label_encoder.transform(test_df["label"].values)
    test_dataset = MentalHealthBERTDataset(
        texts=test_df["clean_text"].tolist(),
        labels=y_test,
        tokenizer=tokenizer,
        max_length=cfg["max_length"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # ── Inference ─────────────────────────────────────────────────────
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Metrics ───────────────────────────────────────────────────────
    class_names = list(label_encoder.classes_)
    accuracy    = accuracy_score(all_labels, all_preds)
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
    )
    macro_f1 = report_dict["macro avg"]["f1-score"]

    logger.info(f"[bert_evaluator] Accuracy : {accuracy:.4f}")
    logger.info(f"[bert_evaluator] Macro F1 : {macro_f1:.4f}")
    logger.info(
        f"[bert_evaluator] Full report:\n"
        + classification_report(all_labels, all_preds, target_names=class_names)
    )

    # ── Save metrics JSON ─────────────────────────────────────────────
    metrics = {
        "model": "bert-base-uncased",
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            cls: {
                "precision": round(report_dict[cls]["precision"], 4),
                "recall":    round(report_dict[cls]["recall"], 4),
                "f1":        round(report_dict[cls]["f1-score"], 4),
            }
            for cls in class_names
        },
    }
    metrics_path = os.path.join(PATHS["metrics_dir"], "bert_metrics.json")
    ensure_dir(PATHS["metrics_dir"])
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"[bert_evaluator] Metrics saved to: {metrics_path}")

    # ── Save confusion matrix PNG ─────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    _save_confusion_matrix(cm, class_names, "bert")

    return metrics


# ══════════════════════════════════════════════════════════════════════
def _save_confusion_matrix(cm, class_names, model_name: str):
    """Render and save confusion matrix as a PNG file."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    figures_dir = PATHS["figures_dir"]
    ensure_dir(figures_dir)
    out_path = os.path.join(figures_dir, f"{model_name}_confusion_matrix.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"[bert_evaluator] Confusion matrix saved to: {out_path}")