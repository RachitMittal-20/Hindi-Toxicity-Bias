#!/usr/bin/env python3
"""
Hindi/Hinglish Toxicity Bias Detection & Mitigation
Full pipeline: data prep → baseline → CDA → adversarial → evaluation

Usage:
    python main.py --stage all
    python main.py --stage baseline
    python main.py --stage cda
    python main.py --stage adversarial
    python main.py --stage evaluate
    python main.py --stage data       # Only run data preparation
"""

import argparse
import sys
from pathlib import Path


def setup_environment():
    """Ensure directories exist."""
    dirs = [
        "data/raw", "data/processed", "data/augmented",
        "models", "results", "logs",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Directory structure ready.")


def check_data():
    """Check if any data files exist in data/raw/."""
    raw_dir = Path("data/raw")
    files = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.tsv"))
    if not files:
        print("\n❌ ERROR: No dataset files found in data/raw/")
        print("   Please download the Kaggle Code-Mixed Hinglish dataset")
        print("   and place the CSV file in data/raw/")
        print("   https://www.kaggle.com/datasets/sharduldhekane/code-mixed-hinglish-hate-speech-detection-dataset")
        sys.exit(1)
    print(f"📂 Found data files: {[f.name for f in files]}")
    return True


def run_data_prep():
    """Run data preparation only."""
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    from src.data_utils import prepare_data
    prepare_data()


def run_baseline():
    """Train baseline XLM-RoBERTa model."""
    print("\n" + "=" * 60)
    print("STAGE 1: Training Baseline Model")
    print("=" * 60)
    from src.train_baseline import main as baseline_main
    baseline_main()


def run_cda():
    """Train CDA-debiased model."""
    print("\n" + "=" * 60)
    print("STAGE 2: Training CDA-Debiased Model")
    print("=" * 60)
    from src.train_cda import main as cda_main
    cda_main()


def run_adversarial():
    """Train adversarially-debiased model."""
    print("\n" + "=" * 60)
    print("STAGE 3: Training Adversarially-Debiased Model")
    print("=" * 60)
    from src.train_adversarial import main as adv_main
    adv_main()


def run_evaluation():
    """Evaluate and compare all models."""
    print("\n" + "=" * 60)
    print("STAGE 4: Evaluating All Models")
    print("=" * 60)
    from src.evaluate import main as eval_main
    eval_main()


def main():
    parser = argparse.ArgumentParser(
        description="Hindi/Hinglish Toxicity Bias Detection & Mitigation"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "data", "baseline", "cda", "adversarial", "evaluate"],
        default="all",
        help="Which stage to run (default: all)",
    )
    args = parser.parse_args()

    setup_environment()
    check_data()

    if args.stage == "data":
        run_data_prep()

    elif args.stage == "all":
        print("\n🚀 Running full pipeline: Data → Baseline → CDA → Adversarial → Evaluation\n")
        run_data_prep()
        run_baseline()
        run_cda()
        run_adversarial()
        run_evaluation()
        print("\n" + "=" * 60)
        print("✅ FULL PIPELINE COMPLETE!")
        print("=" * 60)
        print("Results saved to: results/model_comparison.csv")
        print("Models saved to: models/")

    elif args.stage == "baseline":
        run_baseline()

    elif args.stage == "cda":
        run_cda()

    elif args.stage == "adversarial":
        run_adversarial()

    elif args.stage == "evaluate":
        run_evaluation()

    print("\n✅ Done.\n")


if __name__ == "__main__":
    main()
