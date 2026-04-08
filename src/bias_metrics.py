"""
Bias metrics: FPR Disparity, Demographic Parity Difference, CFT Gap.
Section 3.4 of the paper (Table 5).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_standard_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute accuracy, precision, recall, macro-F1."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def compute_group_fpr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, float]:
    """
    Compute False Positive Rate per identity group.
    FPR(g) = FP_g / (FP_g + TN_g)
    Only computed on truly non-toxic examples (y_true == 0).
    """
    group_fpr = {}
    unique_groups = np.unique(groups)

    for g in unique_groups:
        mask = groups == g
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]

        # Only non-toxic ground truth
        non_toxic_mask = y_true_g == 0
        if non_toxic_mask.sum() == 0:
            continue

        fp = ((y_pred_g[non_toxic_mask] == 1)).sum()
        tn = ((y_pred_g[non_toxic_mask] == 0)).sum()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        group_fpr[str(g)] = fpr

    return group_fpr


def compute_fpr_disparity(group_fpr: Dict[str, float]) -> float:
    """
    FPR-Disparity = max_g FPR(g) - min_g FPR(g)
    Larger values indicate unfair treatment of certain groups.
    """
    if len(group_fpr) < 2:
        return 0.0
    values = list(group_fpr.values())
    return max(values) - min(values)


def compute_demographic_parity_diff(
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> float:
    """
    ΔDP = max_g P(Ŷ=1|g) - min_g P(Ŷ=1|g)
    Measures whether some groups are classified as toxic more often.
    """
    group_positive_rates = {}
    unique_groups = np.unique(groups)

    for g in unique_groups:
        mask = groups == g
        if mask.sum() == 0:
            continue
        positive_rate = y_pred[mask].mean()
        group_positive_rates[str(g)] = positive_rate

    if len(group_positive_rates) < 2:
        return 0.0

    values = list(group_positive_rates.values())
    return max(values) - min(values)


def compute_cft_gap(
    scores_original: np.ndarray,
    scores_counterfactual: np.ndarray,
) -> float:
    """
    CFT Gap = mean(|s(x) - s(x')|)
    where x and x' are counterfactual pairs differing only in identity term.
    """
    gaps = np.abs(scores_original - scores_counterfactual)
    return float(gaps.mean())


class BiasMetricsCalculator:
    """
    Compute all bias metrics for a model's predictions.
    """

    def __init__(self):
        self.results = {}

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        groups: np.ndarray,
        cft_scores_original: Optional[np.ndarray] = None,
        cft_scores_counterfactual: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute all standard + fairness metrics.

        Args:
            y_true: ground truth labels (0/1)
            y_pred: predicted labels (0/1)
            y_scores: predicted toxicity probabilities
            groups: identity group labels per example
            cft_scores_original: toxicity scores on original CFT sentences
            cft_scores_counterfactual: toxicity scores on swapped CFT sentences
        """
        # Standard metrics
        results = compute_standard_metrics(y_true, y_pred)

        # Only compute group metrics on samples WITH identity mentions
        has_group = groups != "none"
        if has_group.sum() > 0:
            y_true_id = y_true[has_group]
            y_pred_id = y_pred[has_group]
            groups_id = groups[has_group]

            # Group FPR
            group_fpr = compute_group_fpr(y_true_id, y_pred_id, groups_id)
            results["group_fpr"] = group_fpr
            results["fpr_disparity"] = compute_fpr_disparity(group_fpr)

            # Demographic parity
            results["demographic_parity_diff"] = compute_demographic_parity_diff(
                y_pred_id, groups_id
            )
        else:
            results["group_fpr"] = {}
            results["fpr_disparity"] = 0.0
            results["demographic_parity_diff"] = 0.0

        # CFT Gap (if counterfactual data provided)
        if cft_scores_original is not None and cft_scores_counterfactual is not None:
            results["cft_gap"] = compute_cft_gap(cft_scores_original, cft_scores_counterfactual)
        else:
            results["cft_gap"] = None

        self.results = results
        return results

    def print_report(self, model_name: str = "Model"):
        """Pretty-print the metrics report."""
        r = self.results
        print(f"\n{'=' * 60}")
        print(f"  {model_name} — Evaluation Report")
        print(f"{'=' * 60}")
        print(f"  Accuracy:   {r['accuracy']:.4f}")
        print(f"  Precision:  {r['precision']:.4f}")
        print(f"  Recall:     {r['recall']:.4f}")
        print(f"  F1:         {r['f1']:.4f}")
        print(f"  Macro-F1:   {r['macro_f1']:.4f}")
        print(f"  ─────────────────────────────────")
        print(f"  FPR Disparity:           {r['fpr_disparity']:.4f}")
        print(f"  Demographic Parity Diff: {r['demographic_parity_diff']:.4f}")
        if r["cft_gap"] is not None:
            print(f"  CFT Gap:                 {r['cft_gap']:.4f}")
        if r["group_fpr"]:
            print(f"  ─────────────────────────────────")
            print(f"  Group-wise FPR:")
            for g, fpr in sorted(r["group_fpr"].items()):
                print(f"    {g:>12s}: {fpr:.4f}")
        print(f"{'=' * 60}\n")
