"""
src/features/bert_dataset.py
────────────────────────────
PyTorch Dataset wrapper for BERT fine-tuning.

Responsibilities:
  - Accept a list of raw text strings and integer labels
  - Tokenize each text using the BERT tokenizer
  - Return input_ids, attention_mask, and label tensors

This module has NO side effects — it is purely a data container
used by trainer.py when use_bert=True.
"""

import torch
from torch.utils.data import Dataset


class MentalHealthBERTDataset(Dataset):
    """
    Wraps tokenized text + labels into a PyTorch Dataset.

    Parameters
    ----------
    texts  : list[str]  — raw (or lightly cleaned) post text
    labels : list[int]  — integer-encoded class labels
    tokenizer          — a HuggingFace PreTrainedTokenizer instance
    max_length : int   — max token sequence length (default 256)
    """

    def __init__(self, texts, labels, tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.texts)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize: pad/truncate to max_length, return PyTorch tensors
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),       # (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_length,)
            "label": torch.tensor(label, dtype=torch.long),
        }