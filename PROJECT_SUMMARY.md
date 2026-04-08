# Hindi–English Code-Mixed Toxicity Bias Mitigation — Complete Project Summary

---

## 1. Problem Statement

Toxicity classifiers for Hindi–English code-mixed (Hinglish) text systematically over-flag benign content that mentions identity terms (religion, caste, gender, region). This produces **disparate false positive rates** across protected subgroups, causing unfair moderation outcomes. The goal: quantify this bias and evaluate two mitigation strategies under matched conditions.

---

## 2. Research Questions

1. Does a standard XLM-RoBERTa fine-tune exhibit measurable identity-correlated bias on Hinglish hate speech data?
2. How does **Counterfactual Data Augmentation (CDA)** compare to **Adversarial Debiasing** for reducing this bias?
3. What is the fairness–utility tradeoff for each method?
4. Do the methods affect *counterfactual fairness* (per-instance robustness to identity swaps), not just group-level statistics?

---

## 3. Dataset

**Source:** Kaggle Code-Mixed Hinglish Hate Speech corpus (`combined_hate_speech_dataset.csv`) — an aggregated public release combining multiple Hindi–English social media sources.

**Note:** The original plan included HASOC, HHSD, and HOLD, but **HASOC access was never granted**, so the experiments use the Kaggle corpus only. Update Table 2 of the paper accordingly.

### Composition

| Split | Samples | Hate (1) | Non-Hate (0) |
|---|---|---|---|
| Train | 20,684 | 9,607 | 11,077 |
| Dev | 2,955 | 1,373 | 1,582 |
| Test | 5,911 | 2,745 | 3,166 |
| **Total** | **29,550** | **13,725** (46.4%) | **15,825** (53.6%) |

- 70 / 10 / 20 stratified split, seed 42
- Auto-detected columns: `text`, `hate_label`
- Reasonably balanced (no severe class imbalance)

### Identity Coverage (lexicon-based, Section 3.2 / Table 3)

| Group | Samples | % of corpus |
|---|---|---|
| Gender | 4,722 | 16.0% |
| Religion | 1,777 | 6.0% |
| Caste | 411 | 1.4% |
| Region | 63 | 0.2% |
| **Any identity** | **6,973** | **23.6%** |
| None | 22,577 | 76.4% |

> ⚠ **Region subgroup (n=63) is too small for reliable bias estimates** — must be flagged in Limitations.

---

## 4. Identity Lexicon

Curated from prior Hinglish hate speech work, four protected attributes:

- **Religion** — Hindu, Muslim, Sikh, Christian, etc. + Hindi/Roman variants
- **Caste** — Brahmin, Dalit, Shudra, jaati terms, etc.
- **Gender** — male/female/trans terms, slurs, identifiers
- **Region** — North/South Indian, state-level identifiers

Identity tags are assigned by string matching against the lexicon during preprocessing.

---

## 5. Methods (three models, identical hyperparameters)

### 5.1 Baseline

- XLM-RoBERTa<sub>base</sub> + linear classification head
- Loss: standard cross-entropy `L_tox`
- No identity signal used during training

### 5.2 CDA (Counterfactual Data Augmentation)

- For each training sample containing an identity term, generate a counterfactual by **swapping the term with another from the same group** (e.g., "Hindu" → "Muslim")
- Train on the union of original + augmented data (1:1 ratio → CAHH corpus)
- Generation is automated in `src/cda.py` during the `cda` stage
- Loss: `L_tox` on the augmented set

### 5.3 Adversarial Debiasing

- Two-head model sharing the XLM-R encoder:
  1. **Toxicity head** → binary toxicity
  2. **Identity head** → 4-way identity group classifier (with **Gradient Reversal Layer**)
- Loss: `L_total = L_tox − λ·L_adv`, with **λ = 0.5**
- Adversary hidden dim = 128
- Gradient reversal pushes the encoder toward identity-invariant representations

---

## 6. Training Configuration (Table 4)

| Parameter | Value |
|---|---|
| Backbone | `xlm-roberta-base` |
| Max sequence length | 128 |
| Batch size | 32 |
| Epochs | 5 |
| Optimizer | AdamW (β₁=0.9, β₂=0.98) |
| Learning rate | 2 × 10⁻⁵ |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Random seed | 42 |
| Best-model selection | Highest dev accuracy |
| Hardware | Apple Silicon MPS (also CUDA/CPU compatible) |

---

## 7. Evaluation Metrics

### Utility

- **Accuracy** — overall classification accuracy
- **Precision / Recall / F1** — for the positive (toxic) class
- **Macro F1** — averaged across both classes

### Fairness (Table 5)

- **FPR Disparity** — max FPR − min FPR across the four identity groups (lower = fairer)
- **Demographic Parity Δ (ΔDP)** — max − min positive prediction rate across groups
- **CFT Gap (Counterfactual Fairness Test Gap)** — measured on a generated CFT test set where each instance is paired with identity-swapped variants; gap = average prediction change after swap

The CFT test set is generated automatically during the CDA stage.

---

## 8. Results

### 8.1 Headline Comparison (test set, n=5,911)

| Model | Acc | Prec | Rec | F1 | Macro F1 | **FPR Δ** ↓ | **DP Δ** ↓ | **CFT Gap** ↓ |
|---|---|---|---|---|---|---|---|---|
| **Baseline** | 0.749 | 0.778 | 0.642 | **0.703** | 0.743 | 0.421 | 0.505 | 0.052 |
| **CDA** | 0.734 | 0.761 | 0.624 | 0.685 | 0.728 | **0.196** | **0.136** | **0.042** |
| **Adversarial** | 0.747 | 0.781 | 0.633 | 0.699 | 0.741 | 0.256 | 0.294 | 0.068 |

**Bold = best per column.**

### 8.2 Relative Improvements (vs Baseline)

| Metric | CDA | Adversarial |
|---|---|---|
| FPR Disparity | **−54%** | −39% |
| Demographic Parity Δ | **−73%** | −42% |
| CFT Gap | **−19%** | **+31%** ⚠ |
| F1 cost | −1.8 pts | −0.4 pts |

### 8.3 Per-Group False Positive Rates

| Group | Baseline | CDA | Adversarial |
|---|---|---|---|
| Religion | 0.2825 | 0.2825 | 0.2486 |
| Caste | 0.4211 | 0.3421 | 0.3947 |
| Gender | 0.1505 | 0.1465 | 0.1386 |
| Region | 0.0000 | 0.1667 | 0.1667 |

(Region values are unreliable due to n=63 — note in paper.)

---

## 9. Key Findings (use these for Discussion)

### Finding 1 — CDA wins on every fairness metric

By directly exposing the model to identity-swapped variants of training examples, CDA breaks the spurious correlation between identity tokens and the toxicity label. The 73% reduction in DP gap and 19% reduction in CFT Gap show this works at *both* the group level and the per-instance level.

### Finding 2 — Adversarial debiasing has a hidden failure mode

Adversarial debiasing improves group-level fairness (FPR Δ, DP Δ) but **worsens** per-instance counterfactual fairness (0.068 vs 0.052 baseline). Gradient reversal produces *shallow decorrelation* — the encoder loses identity information on average, but the decision logic for any single sentence is not made invariant to identity-token edits. This is the most interesting finding and should be the centrepiece of your Discussion.

### Finding 3 — The fairness–utility tradeoff is small and favourable

CDA costs only **1.8 F1 points** for a 54% reduction in FPR disparity. For real moderation deployments where false positives on identity-mentioning content carry harm, this trade is clearly worthwhile.

### Finding 4 — Caste is the most biased subgroup in baseline

Baseline FPR for caste is 0.421 — nearly 3× higher than gender (0.151). CDA reduces this to 0.342. This highlights the importance of caste-aware fairness work for South Asian NLP, which is under-studied.

---

## 10. Limitations (mandatory in paper)

1. **Single dataset.** Results are reported on one Hindi–English corpus; generalisation to other code-mixed languages or domains requires further study. Originally HASOC/HHSD/HOLD were planned but not accessible.
2. **Region subgroup is statistically unreliable.** Only 63 samples (0.2%) — per-group region metrics are reported but should not be interpreted strongly.
3. **Lexicon-based identity detection.** Under-counts implicit references, slang, and informal spellings; introduces selection bias toward explicit identity mentions.
4. **Binary toxicity label.** No distinction between hate speech, profanity, and offensive humour, which limits the granularity of conclusions.
5. **No statistical significance testing.** Single-seed results; multi-seed runs would strengthen the claims.
6. **CFT test set generation is template-based.** Counterfactual quality depends on lexicon coverage and may not produce fully fluent Hinglish.

---

## 11. Reproducibility

### Repo structure

```
hindi-toxicity-bias/
├── configs/training_config.yaml    # all hyperparameters
├── main.py                         # stage orchestrator
├── requirements.txt
├── src/
│   ├── data_utils.py               # CSV auto-detect, preprocessing, splits
│   ├── identity_detection.py       # lexicon + tagging
│   ├── models.py                   # Dataset + adversarial model w/ GRL
│   ├── bias_metrics.py             # FPR Δ, DP Δ, CFT Gap
│   ├── cda.py                      # CAHH corpus + CFT test set generation
│   ├── train_baseline.py
│   ├── train_cda.py
│   ├── train_adversarial.py
│   └── evaluate.py                 # produces results/model_comparison.csv
├── data/{raw,processed,augmented}/
├── models/                         # saved checkpoints
├── results/                        # CSVs with all reported numbers
└── website/                        # standalone HTML preview of results
```

### Five commands to reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python main.py --stage data
.venv/bin/python main.py --stage baseline
.venv/bin/python main.py --stage cda
.venv/bin/python main.py --stage adversarial
.venv/bin/python main.py --stage evaluate
```

### Output files

- `results/model_comparison.csv` — headline table (Section 8.1 numbers)
- `results/group_fpr_details.csv` — per-group FPR breakdown
- `models/hindi_toxicity_baseline/` — best baseline checkpoint
- `models/hindi_toxicity_cda/` — best CDA checkpoint
- `models/hindi_toxicity_adversarial/` — best adversarial checkpoint
- `data/processed/{train,dev,test}.csv` — splits with identity tags
- `data/augmented/cahh_train.csv` — CDA-augmented training set
- `data/augmented/cft_test.csv` — counterfactual fairness test set

---

## 12. Paper Sections — What to Update

| Section | Action |
|---|---|
| **Abstract** | Lead with: "CDA reduces FPR disparity by 54% and DP gap by 73% with only 1.8 F1 cost; adversarial debiasing improves group fairness but worsens counterfactual fairness." |
| **Introduction** | Motivate caste & religion bias in Hinglish moderation; cite the 0.421 baseline caste FPR. |
| **Related Work** | XLM-R, CDA (Lu et al.), adversarial debiasing (Beutel et al., Zhang et al.), Hinglish hate speech work. |
| **Table 2 (Datasets)** | **Remove HASOC/HHSD/HOLD; list only Kaggle Code-Mixed Hinglish corpus**, 29,550 samples. |
| **Section 3.2 (Lexicon)** | Keep as-is — Table 3 unchanged. |
| **Section 3.3 (Methods)** | Keep architecture descriptions; ensure λ=0.5 and aug ratio=1.0 are stated. |
| **Table 4 (Hyperparameters)** | Confirms 128 / 32 / 5 / 2e-5 / 0.1 warmup / 0.01 WD / clip 1.0 / seed 42. |
| **Table 5 (Results)** | Replace with the numbers from Section 8.1 above. |
| **Results section** | Narrate the table; emphasise CDA on all fairness metrics. |
| **Discussion** | Centre on Finding #2 — adversarial debiasing's failure on CFT Gap is the novel insight. |
| **Limitations** | Add Section 10 in full. |
| **Conclusion** | "CDA is the more robust mitigation strategy for code-mixed hate speech; adversarial decorrelation can mask rather than fix identity-driven decisions." |
| **Bibliography** | Add: Conneau et al. 2020 (XLM-R); Lu et al. 2020 (CDA); Ganin & Lempitsky 2015 (GRL); Beutel et al. 2017; Hardt et al. 2016 (equalized odds); Kaggle dataset citation; Bhattacharya et al. or similar Hinglish hate speech surveys. |

---

## 13. The Single Most Quotable Sentence

> "Counterfactual Data Augmentation reduces False Positive Rate Disparity by 54% with a 1.8-point F1 cost, while Adversarial Debiasing improves group-level fairness but **worsens** counterfactual fairness — revealing that gradient reversal can produce shallow decorrelation rather than genuine identity invariance."

That's your headline. Build the paper around it.
