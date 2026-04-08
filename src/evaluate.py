"""
Evaluation pipeline: compare Baseline, CDA, and Adversarial models.
Section 3.6 / Table 6 of the paper.
"""

import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.models import HindiToxicityDataset, AdversarialToxicityModel
from src.data_utils import load_processed
from src.bias_metrics import BiasMetricsCalculator
from src.identity_detection import get_identity_group
from src.train_baseline import get_device


def predict_standard_model(model, dataloader, device):
    """Get predictions and probabilities from a standard HF model."""
    model.eval()
    all_preds = []
    all_scores = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)

            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())  # P(toxic)

    return np.array(all_preds), np.array(all_scores)


def predict_adversarial_model(model, dataloader, device):
    """Get predictions from adversarial model."""
    model.eval()
    all_preds = []
    all_scores = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            tox_logits, _, _ = model(input_ids, attention_mask, lambda_adv=0.0)
            probs = torch.softmax(tox_logits, dim=1)

            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())

    return np.array(all_preds), np.array(all_scores)


def evaluate_on_cft(model, tokenizer, cft_df, device, max_len=128, is_adversarial=False):
    """
    Evaluate model on the CFT Test Set.
    Returns scores for original and counterfactual texts.
    """
    if cft_df is None or len(cft_df) == 0:
        return None, None

    def get_scores(texts):
        dataset = HindiToxicityDataset(texts, [0] * len(texts), tokenizer, max_len)
        loader = DataLoader(dataset, batch_size=64)
        scores = []
        if is_adversarial:
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    tox_logits, _, _ = model(input_ids, attention_mask, lambda_adv=0.0)
                    probs = torch.softmax(tox_logits, dim=1)
                    scores.extend(probs[:, 1].cpu().numpy())
        else:
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=1)
                    scores.extend(probs[:, 1].cpu().numpy())
        return np.array(scores)

    orig_scores = get_scores(cft_df["text_original"].tolist())
    cf_scores = get_scores(cft_df["text_counterfactual"].tolist())
    return orig_scores, cf_scores


def main():
    with open("configs/training_config.yaml") as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"🖥️  Using device: {device}")

    # Load test data
    _, _, test_df = load_processed(config["paths"]["processed_dir"])

    # Ensure identity annotation
    if "identity_group" not in test_df.columns:
        test_df["identity_group"] = test_df["text"].apply(get_identity_group)

    y_true = test_df["label"].values
    groups = test_df["identity_group"].values
    max_len = config["model"]["max_length"]

    # Load CFT test set
    cft_path = f"{config['paths']['augmented_dir']}/cft_test_set.csv"
    cft_df = pd.read_csv(cft_path) if os.path.exists(cft_path) else None
    if cft_df is not None:
        print(f"📂 Loaded CFT test set: {len(cft_df)} pairs")

    all_results = []
    model_dir = config["paths"]["model_dir"]

    # ── Evaluate Baseline ──────────────────────────────────────
    baseline_dir = f"{model_dir}/hindi_toxicity_baseline"
    if os.path.exists(baseline_dir):
        print("\n📊 Evaluating Baseline model...")
        tokenizer = AutoTokenizer.from_pretrained(baseline_dir)
        model = AutoModelForSequenceClassification.from_pretrained(baseline_dir)
        model.to(device)

        test_dataset = HindiToxicityDataset(
            test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_len
        )
        test_loader = DataLoader(test_dataset, batch_size=config["evaluation"]["batch_size"])

        y_pred, y_scores = predict_standard_model(model, test_loader, device)

        cft_orig, cft_cf = evaluate_on_cft(model, tokenizer, cft_df, device, max_len)

        calc = BiasMetricsCalculator()
        results = calc.compute_all(y_true, y_pred, y_scores, groups, cft_orig, cft_cf)
        calc.print_report("Baseline")
        results["model"] = "Baseline"
        all_results.append(results)
    else:
        print(f"⚠️ Baseline model not found at {baseline_dir}")

    # ── Evaluate CDA ───────────────────────────────────────────
    cda_dir = f"{model_dir}/hindi_toxicity_cda"
    if os.path.exists(cda_dir):
        print("\n📊 Evaluating CDA model...")
        tokenizer = AutoTokenizer.from_pretrained(cda_dir)
        model = AutoModelForSequenceClassification.from_pretrained(cda_dir)
        model.to(device)

        test_dataset = HindiToxicityDataset(
            test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_len
        )
        test_loader = DataLoader(test_dataset, batch_size=config["evaluation"]["batch_size"])

        y_pred, y_scores = predict_standard_model(model, test_loader, device)

        cft_orig, cft_cf = evaluate_on_cft(model, tokenizer, cft_df, device, max_len)

        calc = BiasMetricsCalculator()
        results = calc.compute_all(y_true, y_pred, y_scores, groups, cft_orig, cft_cf)
        calc.print_report("CDA")
        results["model"] = "CDA"
        all_results.append(results)
    else:
        print(f"⚠️ CDA model not found at {cda_dir}")

    # ── Evaluate Adversarial ───────────────────────────────────
    adv_dir = f"{model_dir}/hindi_toxicity_adversarial"
    adv_checkpoint = f"{adv_dir}/model.pt"
    if os.path.exists(adv_checkpoint):
        print("\n📊 Evaluating Adversarial model...")
        tokenizer = AutoTokenizer.from_pretrained(adv_dir)

        adv_model = AdversarialToxicityModel(
            base_model_name=config["model"]["base_model"],
            num_labels=config["model"]["num_labels"],
            num_identity_groups=config["adversarial"]["num_identity_groups"],
            adv_hidden_dim=config["adversarial"]["hidden_dim"],
        )
        checkpoint = torch.load(adv_checkpoint, map_location=device)
        adv_model.load_state_dict(checkpoint["model_state_dict"])
        adv_model.to(device)

        test_dataset = HindiToxicityDataset(
            test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_len
        )
        test_loader = DataLoader(test_dataset, batch_size=config["evaluation"]["batch_size"])

        y_pred, y_scores = predict_adversarial_model(adv_model, test_loader, device)

        cft_orig, cft_cf = evaluate_on_cft(
            adv_model, tokenizer, cft_df, device, max_len, is_adversarial=True
        )

        calc = BiasMetricsCalculator()
        results = calc.compute_all(y_true, y_pred, y_scores, groups, cft_orig, cft_cf)
        calc.print_report("Adversarial")
        results["model"] = "Adversarial"
        all_results.append(results)
    else:
        print(f"⚠️ Adversarial model not found at {adv_dir}")

    # ── Save comparison table ──────────────────────────────────
    if all_results:
        results_dir = config["paths"]["results_dir"]
        os.makedirs(results_dir, exist_ok=True)

        summary_rows = []
        for r in all_results:
            summary_rows.append({
                "Model": r["model"],
                "Accuracy": round(r["accuracy"], 4),
                "Precision": round(r["precision"], 4),
                "Recall": round(r["recall"], 4),
                "F1": round(r["f1"], 4),
                "Macro_F1": round(r["macro_f1"], 4),
                "FPR_Disparity": round(r["fpr_disparity"], 4),
                "Demographic_Parity_Diff": round(r["demographic_parity_diff"], 4),
                "CFT_Gap": round(r["cft_gap"], 4) if r["cft_gap"] is not None else "N/A",
            })

        comparison_df = pd.DataFrame(summary_rows)
        comparison_df.to_csv(f"{results_dir}/model_comparison.csv", index=False)

        print("\n" + "=" * 70)
        print("  MODEL COMPARISON SUMMARY")
        print("=" * 70)
        print(comparison_df.to_string(index=False))
        print(f"\n💾 Results saved to {results_dir}/model_comparison.csv")

        # Also save detailed group FPR
        fpr_rows = []
        for r in all_results:
            for group, fpr in r.get("group_fpr", {}).items():
                fpr_rows.append({
                    "Model": r["model"],
                    "Group": group,
                    "FPR": round(fpr, 4),
                })
        if fpr_rows:
            fpr_df = pd.DataFrame(fpr_rows)
            fpr_df.to_csv(f"{results_dir}/group_fpr_details.csv", index=False)
            print(f"💾 Group FPR details saved to {results_dir}/group_fpr_details.csv")


if __name__ == "__main__":
    main()
