"""
Train CDA-debiased model: fine-tune XLM-RoBERTa on original + CAHH augmented data.
Section 3.5.1 of the paper.
"""

import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from src.models import HindiToxicityDataset
from src.data_utils import load_processed
from src.cda import generate_cda_corpus, generate_cft_test_set, save_augmented_data
from src.identity_detection import get_identity_group
from src.train_baseline import train_epoch, evaluate_epoch, get_device


def main():
    with open("configs/training_config.yaml") as f:
        config = yaml.safe_load(f)

    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    print(f"🖥️  Using device: {device}")

    # Load processed data
    train_df, dev_df, test_df = load_processed(config["paths"]["processed_dir"])

    # Generate CDA augmented data (CAHH corpus)
    aug_dir = config["paths"]["augmented_dir"]
    cahh_path = f"{aug_dir}/cahh_corpus.csv"

    if os.path.exists(cahh_path):
        print("📂 Loading existing CAHH corpus...")
        aug_df = pd.read_csv(cahh_path)
    else:
        aug_df = generate_cda_corpus(
            train_df,
            max_augment_ratio=config["cda"]["augment_ratio"],
            seed=seed,
        )
        # Also generate CFT test set while we're at it
        cft_df = generate_cft_test_set(test_df, seed=seed)
        save_augmented_data(aug_df, cft_df, aug_dir)

    # Annotate augmented data with identity info
    aug_df["identity_group"] = aug_df["text"].apply(get_identity_group)
    aug_df["has_identity"] = aug_df["identity_group"] != "none"

    # Combine original + augmented training data
    combined_train = pd.concat([train_df, aug_df[["text", "label"]]], ignore_index=True)
    combined_train = combined_train.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\n📊 Combined training data:")
    print(f"  Original: {len(train_df)}, Augmented: {len(aug_df)}, Total: {len(combined_train)}")

    # Tokenizer & datasets
    model_name = config["model"]["base_model"]
    max_len = config["model"]["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = HindiToxicityDataset(
        combined_train["text"].tolist(), combined_train["label"].tolist(),
        tokenizer, max_len,
    )
    dev_dataset = HindiToxicityDataset(
        dev_df["text"].tolist(), dev_df["label"].tolist(),
        tokenizer, max_len,
    )

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config["evaluation"]["batch_size"])

    # Model (fresh from pretrained, not from baseline checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=config["model"]["num_labels"]
    )
    model.to(device)

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
    save_dir = f"{config['paths']['model_dir']}/hindi_toxicity_cda"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n🚀 Training CDA model for {num_epochs} epochs...")
    print(f"   Train samples: {len(train_dataset)}, Dev samples: {len(dev_dataset)}")

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

    print(f"\n✅ CDA training complete. Best dev accuracy: {best_dev_acc:.4f}")
    print(f"   Model saved to: {save_dir}/")


if __name__ == "__main__":
    main()
