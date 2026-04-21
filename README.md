# Quantifying Bias in Hindi/Hinglish Toxicity Classification

> A controlled study of identity-correlated bias in Hindi–English code-mixed
> hate speech classifiers, comparing **Counterfactual Data Augmentation** and
> **Adversarial Debiasing** on XLM-RoBERTa.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/transformers-4.35%2B-yellow.svg)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/license-Research-lightgrey.svg)](#license)

---

## Authors

**Saurabh Gupta · Devansh Singh · Aviral Chandra · Rachit Mittal · Chanakya Nath**

Department of Computer Science and Engineering
SRM Institute of Science and Technology, Uttar Pradesh, India — 201204

📄 **[Read the full paper (PDF)](Quantifying%20Bias%20in%20Hindi-Hinglish%20Toxicity%20Classification.pdf)**
🌐 **[Project website](https://research.devanshsingh.dev)** · 📊 **[Project summary](PROJECT_SUMMARY.md)**

---

## TL;DR

Toxicity classifiers for Hinglish text systematically over-flag content that
mentions identity terms. We audit a standard XLM-RoBERTa fine-tune and compare
two mitigation strategies under matched conditions:

| Model | F1 | FPR Disparity ↓ | DP Δ ↓ | CFT Gap ↓ |
|---|---|---|---|---|
| Baseline | **0.703** | 0.421 | 0.505 | 0.052 |
| **CDA** | 0.685 | **0.196** (−54%) | **0.136** (−73%) | **0.042** (−19%) |
| Adversarial | 0.699 | 0.256 (−39%) | 0.294 (−42%) | 0.068 (+31%) |

> **Key finding:** CDA reduces FPR disparity by 54% with only 1.8 F1 cost.
> Adversarial debiasing improves group-level fairness but **worsens**
> counterfactual fairness, suggesting gradient reversal can produce shallow
> decorrelation rather than genuine identity invariance.

---

## Repository Structure

```
hindi-toxicity-bias/
├── configs/
│   └── training_config.yaml      # all hyperparameters (Table 4 of paper)
├── data/
│   ├── raw/                      # place Kaggle CSV here
│   ├── processed/                # generated splits with identity tags
│   └── augmented/                # CAHH corpus + CFT test set
├── models/                       # saved checkpoints (gitignored)
├── results/
│   ├── model_comparison.csv      # headline numbers
│   └── group_fpr_details.csv     # per-group FPR breakdown
├── src/
│   ├── data_utils.py             # CSV auto-detect, preprocessing, splits
│   ├── identity_detection.py     # lexicon + tagging (Table 3)
│   ├── models.py                 # Dataset + adversarial model w/ GRL
│   ├── bias_metrics.py           # FPR Δ, DP Δ, CFT Gap
│   ├── cda.py                    # CAHH corpus + CFT test set generation
│   ├── train_baseline.py         # standard fine-tune
│   ├── train_cda.py              # CDA training loop
│   ├── train_adversarial.py      # gradient-reversal training
│   └── evaluate.py               # produces results/model_comparison.csv
├── website/                      # static site preview of results
│   ├── index.html
│   ├── styles.css
│   ├── script.js
│   └── paper.pdf
├── main.py                       # stage orchestrator
├── requirements.txt
├── PROJECT_SUMMARY.md            # detailed write-up of findings
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd hindi-toxicity-bias
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add the dataset

Download the Kaggle Code-Mixed Hinglish Hate Speech corpus and place the CSV
file inside `data/raw/`:

```
data/raw/combined_hate_speech_dataset.csv
```

The data loader auto-detects the text and label columns, so any reasonably
named CSV/TSV will work.

### 3. Run the full pipeline

```bash
python main.py --stage data          # preprocess + split
python main.py --stage baseline      # train XLM-R baseline
python main.py --stage cda           # generate CAHH + train CDA model
python main.py --stage adversarial   # train adversarial model
python main.py --stage evaluate      # compare all 3 models
```

After the final stage you will find:

- `results/model_comparison.csv` — headline metrics table
- `results/group_fpr_details.csv` — per-group FPR breakdown
- `models/hindi_toxicity_*/` — best checkpoints for each method

---

## Methods

Three models are trained on the same data with identical hyperparameters; only
the objective changes.

| Model | Objective | Key idea |
|---|---|---|
| **Baseline** | `L_tox` | Standard XLM-R fine-tune, no fairness intervention |
| **CDA** | `L_tox` on augmented set | Swap identity terms within group; train on union (1:1 ratio) |
| **Adversarial** | `L_total = L_tox − λ·L_adv` | Gradient reversal pushes encoder to be identity-invariant (λ = 0.5) |

### Training Configuration (Table 4)

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

---

## Dataset

| Split | Samples | Hate | Non-Hate |
|---|---|---|---|
| Train | 20,684 | 9,607 | 11,077 |
| Dev | 2,955 | 1,373 | 1,582 |
| Test | 5,911 | 2,745 | 3,166 |
| **Total** | **29,550** | 13,725 (46.4%) | 15,825 (53.6%) |

Identity-bearing samples (lexicon-detected): **6,973 (23.6%)** — gender 4,722 ·
religion 1,777 · caste 411 · region 63.

> ⚠ The region subgroup (n=63) is too small for reliable bias estimates and
> is reported but not interpreted strongly.

---

## Evaluation Metrics

**Utility:** Accuracy · Precision · Recall · F1 · Macro F1
**Fairness:**
- **FPR Disparity** — `max FPR − min FPR` across identity groups (lower is fairer)
- **Demographic Parity Δ** — `max − min` positive prediction rate across groups
- **CFT Gap** — average prediction change on identity-swapped sentence pairs

The CFT test set is generated automatically during the `cda` stage.

---

## Project Website

A self-contained static site is available in [`website/`](website/) with
interactive results, animated charts, author bios, and a direct PDF link:

```bash
cd website && python3 -m http.server 8000
# open http://localhost:8000
```

---

## Hardware

The pipeline supports CUDA, Apple Silicon MPS, and CPU. Approximate end-to-end
training time:

| Hardware | Baseline | CDA | Adversarial | Total |
|---|---|---|---|---|
| Apple M-series (MPS) | ~45 min | ~75 min | ~50 min | ~3 hr |
| NVIDIA T4 | ~12 min | ~22 min | ~14 min | ~50 min |
| NVIDIA A100 | ~4 min | ~7 min | ~5 min | ~16 min |

---

## Citation

If you use this code or the released CFT test set in your work, please cite:

```bibtex
@misc{gupta2026hinditoxicitybias,
  title  = {Quantifying Bias in Hindi/Hinglish Toxicity Classification},
  author = {Gupta, Saurabh and Singh, Devansh and Chandra, Aviral and
            Mittal, Rachit and Nath, Chanakya},
  year   = {2026},
  institution = {SRM Institute of Science and Technology},
  note   = {Department of Computer Science and Engineering}
}
```

---

## Limitations

1. Single dataset — generalisation to other code-mixed languages requires further study
2. Region subgroup is statistically unreliable (n=63)
3. Lexicon-based identity detection under-counts implicit references and informal spellings
4. Binary toxicity label — no distinction between hate speech, profanity, and offensive humour
5. Single-seed results — multi-seed runs would strengthen the claims
6. Template-based counterfactual generation may not produce fully fluent Hinglish

See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for the complete discussion.

---

## Contact

For questions, dataset access, or collaboration enquiries:

| Author | Email |
|---|---|
| Saurabh Gupta | [saurabhg1@srmist.edu.in](mailto:saurabhg1@srmist.edu.in) |
| Devansh Singh | [ds2553@srmist.edu.in](mailto:ds2553@srmist.edu.in) |
| Aviral Chandra | [ac5379@srmist.edu.in](mailto:ac5379@srmist.edu.in) |
| Rachit Mittal | [rm8782@srmist.edu.in](mailto:rm8782@srmist.edu.in) |
| Chanakya Nath | [cn2211@srmist.edu.in](mailto:cn2211@srmist.edu.in) |

---

## License

Released for academic and research use. Please cite the paper if you build on
this work. Dataset usage is subject to the original Kaggle Code-Mixed Hinglish
Hate Speech corpus license.
