"""
Train adversarially-debiased model.
Section 3.5.2: L_total = L_tox - lambda * L_adv
"""

import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.models import AdversarialToxicityModel, HindiToxicityDatasetWithGroup
from src.data_utils import load_processed
from src.identity_detection import get_identity_group
from src.train_baseline import get_device


# Map identity group strings to integer labels
GROUP_TO_ID = {"religion": 0, "caste": 1, "gender": 2, "region": 3, "none": 4}


def encode_groups(groups: list) -> list:
    return [GROUP_TO_ID.get(g, 4) for g in groups]


def train_adversarial_epoch(model, dataloader, optimizer, scheduler, device,
                            lambda_adv=0.5, grad_clip=1.0):
    model.train()
    tox_criterion = nn.CrossEntropyLoss()
    adv_criterion = nn.CrossEntropyLoss()

    total_tox_loss = 0
    total_adv_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        groups = batch["group"].to(device)

        optimizer.zero_grad()

        tox_logits, adv_logits, _ = model(
            input_ids, attention_mask, lambda_adv=lambda_adv
        )

        tox_loss = tox_criterion(tox_logits, labels)
        adv_loss = adv_criterion(adv_logits, groups)

        # Combined loss: gradient reversal handles the negative sign for adversary
        loss = tox_loss + adv_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_tox_loss += tox_loss.item() * input_ids.size(0)
        total_adv_loss += adv_loss.item() * input_ids.size(0)
        preds = torch.argmax(tox_logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    n = total
    return total_tox_loss / n, total_adv_loss / n, correct / n


def evaluate_adversarial_epoch(model, dataloader, device, lambda_adv=0.5):
    model.eval()
    tox_criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            tox_logits, _, _ = model(input_ids, attention_mask, lambda_adv=lambda_adv)
            loss = tox_criterion(tox_logits, labels)

            total_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(tox_logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    with open("configs/training_config.yaml") as f:
        config = yaml.safe_load(f)

    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    print(f"🖥️  Using device: {device}")

    # Load data
    train_df, dev_df, test_df = load_processed(config["paths"]["processed_dir"])

    # Ensure identity_group column exists
    if "identity_group" not in train_df.columns:
        train_df["identity_group"] = train_df["text"].apply(get_identity_group)
    if "identity_group" not in dev_df.columns:
        dev_df["identity_group"] = dev_df["text"].apply(get_identity_group)

    # Encode groups
    train_groups = encode_groups(train_df["identity_group"].tolist())
    dev_groups = encode_groups(dev_df["identity_group"].tolist())

    # Tokenizer & datasets
    model_name = config["model"]["base_model"]
    max_len = config["model"]["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = HindiToxicityDatasetWithGroup(
        train_df["text"].tolist(), train_df["label"].tolist(),
        train_groups, tokenizer, max_len,
    )
    dev_dataset = HindiToxicityDatasetWithGroup(
        dev_df["text"].tolist(), dev_df["label"].tolist(),
        dev_groups, tokenizer, max_len,
    )

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config["evaluation"]["batch_size"])

    # Model
    model = AdversarialToxicityModel(
        base_model_name=model_name,
        num_labels=config["model"]["num_labels"],
        num_identity_groups=config["adversarial"]["num_identity_groups"],
        adv_hidden_dim=config["adversarial"]["hidden_dim"],
    )
    model.to(device)

    lambda_adv = config["adversarial"]["lambda_adv"]

    # Optimizer & scheduler
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

    # Train
    best_dev_acc = 0
    save_dir = f"{config['paths']['model_dir']}/hindi_toxicity_adversarial"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n🚀 Training adversarial model for {num_epochs} epochs (λ_adv={lambda_adv})...")
    print(f"   Train samples: {len(train_dataset)}, Dev samples: {len(dev_dataset)}")

    for epoch in range(num_epochs):
        tox_loss, adv_loss, train_acc = train_adversarial_epoch(
            model, train_loader, optimizer, scheduler, device,
            lambda_adv=lambda_adv,
            grad_clip=config["training"]["gradient_clip"],
        )
        dev_loss, dev_acc = evaluate_adversarial_epoch(
            model, dev_loader, device, lambda_adv=lambda_adv,
        )

        print(
            f"  Epoch {epoch+1}/{num_epochs} | "
            f"Tox Loss: {tox_loss:.4f} Adv Loss: {adv_loss:.4f} Acc: {train_acc:.4f} | "
            f"Dev Loss: {dev_loss:.4f} Dev Acc: {dev_acc:.4f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
            }, f"{save_dir}/model.pt")
            tokenizer.save_pretrained(save_dir)
            print(f"    ✅ Saved best model (dev_acc={dev_acc:.4f})")

    print(f"\n✅ Adversarial training complete. Best dev accuracy: {best_dev_acc:.4f}")
    print(f"   Model saved to: {save_dir}/")


if __name__ == "__main__":
    main()
