"""
PyTorch Dataset wrapper and adversarial model architecture.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel


class HindiToxicityDataset(Dataset):
    """PyTorch Dataset for tokenised text + labels."""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class HindiToxicityDatasetWithGroup(Dataset):
    """Dataset that also returns identity group label (for adversarial training)."""

    def __init__(self, texts, labels, groups, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.groups = groups
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        group = int(self.groups[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "group": torch.tensor(group, dtype=torch.long),
        }


class GradientReversal(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class AdversarialToxicityModel(nn.Module):
    """
    Two-head model: toxicity classifier + identity adversary.
    Section 3.5.2 of the paper.

    L_total = L_tox - lambda * L_adv
    The gradient reversal layer handles the negative sign automatically.
    """

    def __init__(self, base_model_name="xlm-roberta-base", num_labels=2,
                 num_identity_groups=4, adv_hidden_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size

        # Toxicity classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )

        # Adversary head (predicts identity group from [CLS])
        self.adversary = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, adv_hidden_dim),
            nn.ReLU(),
            nn.Linear(adv_hidden_dim, num_identity_groups + 1),  # +1 for 'none'
        )

    def forward(self, input_ids, attention_mask, lambda_adv=0.5):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Toxicity logits (normal forward)
        tox_logits = self.classifier(cls_repr)

        # Adversary logits (through gradient reversal)
        reversed_repr = GradientReversal.apply(cls_repr, lambda_adv)
        adv_logits = self.adversary(reversed_repr)

        return tox_logits, adv_logits, cls_repr
