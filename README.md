# Home Credit Default — Class Imbalance Study

A comparative study of **shallow vs deep learning models** under **class imbalance handling techniques** for credit default prediction.

**Course:** DATS 6202 (Machine Learning I), Spring 2026
**Team:** Minwoo Yoo, Nathan
**Instructor:** Prof. Yuxiao Huang, GWU

---

## Problem

Predict the probability that a Home Credit applicant defaults on a loan (binary classification, `TARGET ∈ {0, 1}`). The training set is **imbalanced** — only ~8% of applicants default — which makes naive models prone to favor the majority class.

## Research Question

> *Do shallow and deep models respond differently to class imbalance handling techniques?*

## Experiment Design

A two-axis comparison: **3 model families × 4 imbalance-handling techniques**.

|                    | Baseline | `class_weight` | SMOTE | Threshold tuning |
| ------------------ | :------: | :------------: | :---: | :--------------: |
| Logistic Regression |    ✓    |       ✓        |   ✓   |        ✓         |
| LightGBM            |    ✓    |       ✓        |   ✓   |        ✓         |
| MLP (sklearn)       |    ✓    |       ✓        |   ✓   |        ✓         |

**Validation:** Stratified 5-Fold CV (SMOTE applied inside each fold to avoid leakage).
**Metrics:** ROC AUC (primary), PR AUC, recall/precision/F1.

## Data

- **Source:** [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/competitions/home-credit-default-risk/data)
- **Files used:** `application_train.csv` (307,511 × 122), `application_test.csv` (48,744 × 121)
- **Note:** Data files are not committed to the repo. Download from Kaggle and place under `data/`.

## Repo Layout

```
.
├── notebooks/      # Jupyter notebooks (project notebook = report)
├── data/           # Kaggle CSVs (gitignored)
├── slides/         # Final presentation (with recording link)
├── recording/      # 8-10 min recorded talk (or link)
└── README.md
```

## Reproduce

```bash
pip install scikit-learn lightgbm imbalanced-learn matplotlib seaborn pandas numpy
# place application_train.csv and application_test.csv under data/
jupyter notebook notebooks/
```
