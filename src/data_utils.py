"""
Data loading, preprocessing, and splitting utilities.
Auto-detects column names from the Kaggle Code-Mixed Hinglish dataset.
"""

import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.identity_detection import detect_identity_terms, get_identity_group, has_identity_mention

# ---------------------------------------------------------------------------
# Column name auto-detection
# ---------------------------------------------------------------------------

# Common column name patterns for text and label
TEXT_CANDIDATES = ["text", "tweet", "sentence", "comment", "content", "post", "message"]
LABEL_CANDIDATES = ["label", "class", "category", "hate", "offensive", "toxic", "sentiment", "task_1", "task1"]


def _find_column(df: pd.DataFrame, candidates: List[str], fallback_idx: int = 0) -> str:
    """Find the best matching column name from candidates."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    # Exact match first
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    # Substring match
    for cand in candidates:
        for col_low, col_orig in cols_lower.items():
            if cand in col_low:
                return col_orig
    # Fallback: use positional index
    return df.columns[fallback_idx]


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Light normalisation as described in Section 3.2:
    - Unicode normalisation (NFC)
    - URL removal
    - User handle removal (@user)
    - Whitespace cleanup
    - Lowercasing
    """
    if not isinstance(text, str):
        return ""
    # Unicode NFC normalisation
    text = unicodedata.normalize("NFC", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove @mentions
    text = re.sub(r"@\w+", "", text)
    # Remove hashtag symbols (keep the word)
    text = re.sub(r"#", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Lowercase (for lexicon matching; tokeniser handles casing)
    text = text.lower()
    return text


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_kaggle_dataset(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load the Kaggle Code-Mixed Hinglish Hate Speech Detection Dataset.
    Auto-detects file name and column structure.
    """
    data_path = Path(data_dir)

    # Find CSV/TSV files
    files = list(data_path.glob("*.csv")) + list(data_path.glob("*.tsv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV/TSV files found in {data_dir}. "
            f"Please download the Kaggle dataset and place it in {data_dir}/"
        )

    print(f"📂 Found data files: {[f.name for f in files]}")

    # Load all files and concatenate
    dfs = []
    for f in files:
        sep = "\t" if f.suffix == ".tsv" else ","
        try:
            df = pd.read_csv(f, sep=sep, on_bad_lines="skip", engine="python")
            print(f"  ✅ Loaded {f.name}: {df.shape[0]} rows, columns={df.columns.tolist()}")
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠️ Failed to load {f.name}: {e}")

    if not dfs:
        raise RuntimeError("Could not load any data files.")

    df = pd.concat(dfs, ignore_index=True)

    # Auto-detect columns
    text_col = _find_column(df, TEXT_CANDIDATES, fallback_idx=0)
    label_col = _find_column(df, LABEL_CANDIDATES, fallback_idx=-1)

    print(f"\n🔍 Auto-detected columns:")
    print(f"  Text column:  '{text_col}'")
    print(f"  Label column: '{label_col}'")

    # Rename to standard names
    df = df.rename(columns={text_col: "text", label_col: "label"})

    # Drop rows with missing text or label
    df = df.dropna(subset=["text", "label"])

    # Normalise labels to binary (0 = non-toxic, 1 = toxic)
    df["label"] = _normalise_labels(df["label"])
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)

    print(f"\n📊 Dataset summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

    return df


def _normalise_labels(labels: pd.Series) -> pd.Series:
    """
    Map various label formats to binary 0/1.
    Handles: numeric (0/1), string ('hate'/'normal'), etc.
    """
    # If already numeric
    if pd.api.types.is_numeric_dtype(labels):
        return labels.astype(int)

    # String labels - map common patterns
    label_map = {}
    unique_labels = labels.dropna().unique()
    print(f"  Raw label values: {unique_labels}")

    for lbl in unique_labels:
        lbl_lower = str(lbl).lower().strip()
        if lbl_lower in ["hate", "hate speech", "hateful", "offensive", "toxic",
                         "hatespeech", "hate_speech", "hot", "1", "yes",
                         "abusive", "hostile"]:
            label_map[lbl] = 1
        elif lbl_lower in ["normal", "non-hate", "non-offensive", "not",
                           "nothate", "not_hate", "none", "0", "no",
                           "non-toxic", "benign", "normal speech"]:
            label_map[lbl] = 0
        else:
            # Unknown - try numeric conversion
            try:
                label_map[lbl] = int(float(lbl_lower))
            except (ValueError, TypeError):
                print(f"  ⚠️ Unknown label '{lbl}' - mapping to 1 (toxic)")
                label_map[lbl] = 1

    print(f"  Label mapping: {label_map}")
    return labels.map(label_map)


# ---------------------------------------------------------------------------
# Identity annotation
# ---------------------------------------------------------------------------

def annotate_identity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add identity annotation columns to the dataframe.
    - identity_terms: dict of {category: [terms]}
    - identity_group: primary identity category
    - has_identity: boolean flag
    """
    df = df.copy()
    df["identity_terms"] = df["text"].apply(detect_identity_terms)
    df["identity_group"] = df["text"].apply(get_identity_group)
    df["has_identity"] = df["text"].apply(has_identity_mention)

    # Report coverage
    n_identity = df["has_identity"].sum()
    pct = 100 * n_identity / len(df)
    print(f"\n🏷️  Identity annotation:")
    print(f"  Texts with identity terms: {n_identity}/{len(df)} ({pct:.1f}%)")
    print(f"  Group distribution:")
    print(f"{df['identity_group'].value_counts().to_string()}")

    return df


# ---------------------------------------------------------------------------
# Train / Dev / Test splits
# ---------------------------------------------------------------------------

def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split into train/dev/test (70/10/20 as per paper Section 3.1).
    Stratified on label to maintain class balance.
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6

    # First split: train vs (dev+test)
    train_df, temp_df = train_test_split(
        df, test_size=(dev_ratio + test_ratio),
        stratify=df["label"], random_state=seed,
    )

    # Second split: dev vs test
    relative_test = test_ratio / (dev_ratio + test_ratio)
    dev_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        stratify=temp_df["label"], random_state=seed,
    )

    print(f"\n📐 Data splits:")
    print(f"  Train: {len(train_df)} (label dist: {dict(train_df['label'].value_counts())})")
    print(f"  Dev:   {len(dev_df)} (label dist: {dict(dev_df['label'].value_counts())})")
    print(f"  Test:  {len(test_df)} (label dist: {dict(test_df['label'].value_counts())})")

    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Save / load processed data
# ---------------------------------------------------------------------------

def save_processed(train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame,
                   output_dir: str = "data/processed"):
    """Save processed splits to disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    dev_df.to_csv(f"{output_dir}/dev.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    print(f"💾 Saved processed data to {output_dir}/")


def load_processed(processed_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously processed splits."""
    train_df = pd.read_csv(f"{processed_dir}/train.csv")
    dev_df = pd.read_csv(f"{processed_dir}/dev.csv")
    test_df = pd.read_csv(f"{processed_dir}/test.csv")
    return train_df, dev_df, test_df


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def prepare_data(data_dir: str = "data/raw", output_dir: str = "data/processed",
                 seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end data preparation:
    1. Load raw dataset
    2. Preprocess text
    3. Annotate identity terms
    4. Create train/dev/test splits
    5. Save to disk
    """
    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)

    # Load
    df = load_kaggle_dataset(data_dir)

    # Preprocess
    print("\n🔧 Preprocessing text...")
    df["text_original"] = df["text"]
    df["text"] = df["text"].apply(preprocess_text)

    # Remove empty texts after preprocessing
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    print(f"  Samples after preprocessing: {len(df)}")

    # Annotate identity
    df = annotate_identity(df)

    # Split
    train_df, dev_df, test_df = create_splits(df, seed=seed)

    # Save
    save_processed(train_df, dev_df, test_df, output_dir)

    return train_df, dev_df, test_df


if __name__ == "__main__":
    prepare_data()
