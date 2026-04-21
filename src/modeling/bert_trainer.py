"""
src/modeling/bert_trainer.py
────────────────────────────
Fine-tunes bert-base-uncased for 5-class mental health classification.

Architecture decision
─────────────────────
  - bert-base-uncased  →  BertForSequenceClassification (num_labels=5)
  - The model's built-in classification head replaces the [CLS] pooler
    output with a linear layer: hidden_size → num_labels.
  - Loss: CrossEntropyLoss (computed internally by the HuggingFace model)
  - Optimizer: AdamW with weight decay

Separation of concerns (matches SYSTEM_FLOW.md rules)
──────────────────────────────────────────────────────
  - This module ONLY trains and saves. It does NOT evaluate.
  - Evaluation is handled by bert_evaluator.py / evaluator.py
  - All hyperparameters come from config.py
"""

import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder

# ── project-local imports ──────────────────────────────────────────────
from src.features.bert_dataset import MentalHealthBERTDataset
from src.utils.config import BERT_CONFIG
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
def train_bert(train_df, val_df, label_encoder: LabelEncoder):
    """
    Fine-tune BERT on mental health Reddit posts.

    Parameters
    ----------
    train_df      : pd.DataFrame with 'clean_text' and 'label' columns
    val_df        : pd.DataFrame with 'clean_text' and 'label' columns
    label_encoder : fitted sklearn LabelEncoder (integer ↔ class name)

    Returns
    -------
    model     : fine-tuned BertForSequenceClassification (on CPU/GPU)
    tokenizer : the BertTokenizer used during training
    """
    cfg = BERT_CONFIG          # shorthand
    device = _get_device()
    logger.info(f"[bert_trainer] Using device: {device}")

    # ── 1. Tokenizer ──────────────────────────────────────────────────
    logger.info(f"[bert_trainer] Loading tokenizer: {cfg['model_name']}")
    tokenizer = BertTokenizer.from_pretrained(cfg["model_name"])

    # ── 2. Encode labels to integers ──────────────────────────────────
    y_train = label_encoder.transform(train_df["label"].values)
    y_val   = label_encoder.transform(val_df["label"].values)

    # ── 3. Build Datasets & DataLoaders ───────────────────────────────
    train_dataset = MentalHealthBERTDataset(
        texts=train_df["clean_text"].tolist(),
        labels=y_train,
        tokenizer=tokenizer,
        max_length=cfg["max_length"],
    )
    val_dataset = MentalHealthBERTDataset(
        texts=val_df["clean_text"].tolist(),
        labels=y_val,
        tokenizer=tokenizer,
        max_length=cfg["max_length"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,          # keep 0 for Windows/macOS compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # ── 4. Model ──────────────────────────────────────────────────────
    num_labels = len(label_encoder.classes_)
    logger.info(f"[bert_trainer] Initialising model with {num_labels} output classes")
    model = BertForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=num_labels,
    )
    model.to(device)

    # ── 5. Optimiser & LR scheduler ───────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── 6. Training loop ──────────────────────────────────────────────
    logger.info(f"[bert_trainer] Starting training for {cfg['epochs']} epoch(s)")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()

            # Forward pass — loss is computed inside the HuggingFace model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            # Gradient clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            if step % cfg["log_every_n_steps"] == 0:
                avg = total_train_loss / step
                logger.info(
                    f"  Epoch {epoch}/{cfg['epochs']}  |  "
                    f"Step {step}/{len(train_loader)}  |  "
                    f"Avg train loss: {avg:.4f}"
                )

        # ── Validation loss at end of each epoch ─────────────────────
        val_loss = _compute_val_loss(model, val_loader, device)
        logger.info(
            f"[bert_trainer] ✓ Epoch {epoch} complete  "
            f"| val_loss: {val_loss:.4f}"
        )

    # ── 7. Save model + tokenizer ─────────────────────────────────────
    save_dir = cfg["save_dir"]
    ensure_dir(save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"[bert_trainer] Model + tokenizer saved to: {save_dir}")

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════

def _get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_val_loss(model, val_loader, device) -> float:
    """Run one pass over the validation set and return average loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["label"].to(device),
            )
            total_loss += outputs.loss.item()
    return total_loss / len(val_loader)