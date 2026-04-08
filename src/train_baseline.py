"""
Train baseline XLM-RoBERTa toxicity classifier.
Section 3.3 / Table 4 of the paper.
"""

import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from src.models import HindiToxicityDataset
from src.data_utils import prepare_data, load_processed


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            total_loss += outputs.loss.item() * input_ids.size(0)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    # Load config
    with open("configs/training_config.yaml") as f:
        config = yaml.safe_load(f)

    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    print(f"🖥️  Using device: {device}")

    # Prepare data (or load if already processed)
    processed_dir = config["paths"]["processed_dir"]
    if os.path.exists(f"{processed_dir}/train.csv"):
        print("📂 Loading previously processed data...")
        from src.data_utils import load_processed
        train_df, dev_df, test_df = load_processed(processed_dir)
    else:
        train_df, dev_df, test_df = prepare_data(
            data_dir=config["paths"]["data_dir"],
            output_dir=processed_dir,
            seed=seed,
        )

    # Tokenizer
    model_name = config["model"]["base_model"]
    max_len = config["model"]["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Datasets
    train_dataset = HindiToxicityDataset(
        train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_len
    )
    dev_dataset = HindiToxicityDataset(
        dev_df["text"].tolist(), dev_df["label"].tolist(), tokenizer, max_len
    )

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config["evaluation"]["batch_size"])

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=config["model"]["num_labels"]
    )
    model.to(device)

    # Optimizer & scheduler (Table 4)
    num_epochs = config["training"]["num_epochs"]
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=(0.9, 0.98),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    best_dev_acc = 0
    save_dir = f"{config['paths']['model_dir']}/hindi_toxicity_baseline"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n🚀 Training baseline model for {num_epochs} epochs...")
    print(f"   Train samples: {len(train_dataset)}, Dev samples: {len(dev_dataset)}")
    print(f"   Batch size: {batch_size}, LR: {config['training']['learning_rate']}")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_clip=config["training"]["gradient_clip"],
        )
        dev_loss, dev_acc = evaluate_epoch(model, dev_loader, device)

        print(
            f"  Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Dev Loss: {dev_loss:.4f} Acc: {dev_acc:.4f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"    ✅ Saved best model (dev_acc={dev_acc:.4f})")

    print(f"\n✅ Baseline training complete. Best dev accuracy: {best_dev_acc:.4f}")
    print(f"   Model saved to: {save_dir}/")


if __name__ == "__main__":
    main()
