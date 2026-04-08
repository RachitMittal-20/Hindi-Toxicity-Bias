"""
Counterfactual Data Augmentation (CDA) — Section 3.5.1
Also generates the CFT Test Set (counterfactual paired test set).
"""

import re
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.identity_detection import (
    IDENTITY_LEXICON,
    detect_identity_terms,
    get_swap_pairs,
)


def _build_swap_map() -> Dict[str, List[str]]:
    """
    Build a flat mapping: term -> list of replacement terms from same category.
    E.g., 'hindu' -> ['muslim', 'christian', 'sikh', ...]
    """
    swap_pairs = get_swap_pairs()
    swap_map: Dict[str, List[str]] = {}

    for category, groups in swap_pairs.items():
        # Flatten each group to its first (canonical) term for swapping
        canonical = [g[0] for g in groups]
        # Also map all variants
        for group in groups:
            for term in group:
                others = [c for c in canonical if c != group[0]]
                swap_map[term.lower()] = others

    return swap_map


SWAP_MAP = _build_swap_map()


def augment_single(text: str, label: int) -> List[Tuple[str, int]]:
    """
    Generate counterfactual augmentations for a single text.
    For each identity term found, swap it with other terms from the same category.
    Preserves the original label.

    Returns list of (augmented_text, label) tuples.
    """
    augmented = []
    text_lower = text.lower()

    for term, replacements in SWAP_MAP.items():
        if term in text_lower:
            for repl in replacements:
                # Case-insensitive replacement
                new_text = re.sub(
                    re.escape(term),
                    repl,
                    text_lower,
                    flags=re.IGNORECASE,
                )
                if new_text != text_lower:
                    augmented.append((new_text, label))

    return augmented


def generate_cda_corpus(
    train_df: pd.DataFrame,
    max_augment_ratio: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate the CAHH (Counterfactual Augmented Hindi/Hinglish) corpus.
    Section 3.5.1 of the paper.

    For each training example containing an identity term, generate
    augmented examples by swapping identity terms.

    Args:
        train_df: training dataframe with 'text' and 'label' columns
        max_augment_ratio: max ratio of augmented to original samples
        seed: random seed

    Returns:
        augmented_df: new augmented examples (not including originals)
    """
    random.seed(seed)
    augmented_rows = []

    identity_count = 0
    for _, row in train_df.iterrows():
        text, label = row["text"], row["label"]
        augs = augment_single(text, label)
        if augs:
            identity_count += 1
            augmented_rows.extend(augs)

    # Cap to max_augment_ratio
    max_samples = int(len(train_df) * max_augment_ratio)
    if len(augmented_rows) > max_samples:
        augmented_rows = random.sample(augmented_rows, max_samples)

    aug_df = pd.DataFrame(augmented_rows, columns=["text", "label"])

    print(f"\n🔄 CDA Augmentation:")
    print(f"  Training samples with identity terms: {identity_count}")
    print(f"  Total augmented samples generated: {len(aug_df)}")
    print(f"  Augmented label distribution:\n{aug_df['label'].value_counts().to_string()}")

    return aug_df


def generate_cft_test_set(
    test_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate the CFT Test Set — paired counterfactual test examples.
    Section 3.1 / contribution #2.

    For each test example with an identity term, create paired versions
    where only the identity term is swapped.

    Returns DataFrame with columns:
        text_original, text_counterfactual, label, swap_category, term_original, term_swapped
    """
    random.seed(seed)
    pairs = []

    for _, row in test_df.iterrows():
        text = row["text"]
        label = row["label"]
        text_lower = text.lower()

        for term, replacements in SWAP_MAP.items():
            if term in text_lower and replacements:
                repl = random.choice(replacements)
                swapped = re.sub(re.escape(term), repl, text_lower, flags=re.IGNORECASE)
                if swapped != text_lower:
                    pairs.append({
                        "text_original": text_lower,
                        "text_counterfactual": swapped,
                        "label": label,
                        "term_original": term,
                        "term_swapped": repl,
                    })

    cft_df = pd.DataFrame(pairs)
    print(f"\n🔀 CFT Test Set:")
    print(f"  Total counterfactual pairs: {len(cft_df)}")
    if len(cft_df) > 0:
        print(f"  Label distribution:\n{cft_df['label'].value_counts().to_string()}")

    return cft_df


def save_augmented_data(aug_df: pd.DataFrame, cft_df: pd.DataFrame,
                        output_dir: str = "data/augmented"):
    """Save augmented training data and CFT test set."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    aug_df.to_csv(f"{output_dir}/cahh_corpus.csv", index=False)
    cft_df.to_csv(f"{output_dir}/cft_test_set.csv", index=False)
    print(f"💾 Saved CAHH corpus and CFT test set to {output_dir}/")


if __name__ == "__main__":
    # Quick test
    test_texts = [
        ("those muslims are criminals", 1),
        ("hindu festival is beautiful", 0),
        ("दलित लोग मेहनती हैं", 0),
    ]
    for text, label in test_texts:
        augs = augment_single(text, label)
        print(f"Original: {text}")
        for a_text, a_label in augs[:3]:
            print(f"  -> {a_text} (label={a_label})")
        print()
