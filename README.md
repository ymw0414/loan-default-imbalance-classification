# Home Credit Default — Class Imbalance Study

A comparative study of **classic shallow learners** and **neural networks** for credit default prediction, with a focused look at how each model family handles class imbalance.

## Problem

Predict the probability that a Home Credit applicant defaults on a loan (binary classification, `TARGET ∈ {0, 1}`). The training set is **heavily imbalanced** — only ~8% of applicants default — so the default 0.5 decision rule routinely produces a 0-recall classifier even when the underlying ranker has decent ROC AUC.

## Research Question

> *Do classic shallow learners and neural networks respond differently to class imbalance handling techniques?*

## Models Compared

| Model | Family | Owner | Imbalance handling |
| :--- | :--- | :--- | :--- |
| Logistic Regression  | Linear           | Nathaniel Badalov | `class_weight='balanced'` |
| Random Forest        | Tree (bagging)   | Nathaniel Badalov | `class_weight='balanced'` |
| HistGradientBoosting | Tree (boosting)  | Nathaniel Badalov | tuned hyperparameters    |
| Shallow MLP `(64,)`         | Neural network | Minwoo Yoo | baseline / SMOTE / threshold tuning |
| Deep MLP `(128, 64, 32)`    | Neural network | Minwoo Yoo | baseline / SMOTE / threshold tuning |

**Validation:** 80/20 stratified train–val split + `GridSearchCV` with `PredefinedSplit`, scored on ROC AUC.

## Headline Results (validation ROC AUC)

| Rank | Model | val ROC AUC |
| :---: | :--- | :---: |
| 1 | HistGradientBoosting | **0.7595** |
| 2 | Logistic Regression  | 0.7487 |
| 3 | Random Forest        | 0.7470 |
| 4 | Deep MLP             | 0.7416 |
| 5 | Shallow MLP          | 0.7405 |

Imbalance comparison on the best MLP:

| Method | Threshold | ROC AUC | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline           | 0.500 | 0.745 | 0.00 | 0.00 | 0.00 |
| SMOTE oversampling | 0.500 | 0.648 | 0.16 | 0.21 | 0.18 |
| Threshold tuning   | 0.164 | 0.745 | 0.24 | 0.40 | 0.30 |

> Threshold tuning preserves AUC and lifts recall from 0% to ~40%; SMOTE moved AUC in the wrong direction. *Choosing the threshold matters more than rebalancing the data here.*

## Data

- **Source:** [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/competitions/home-credit-default-risk/data)
- **Files used:** `application_train.csv` (307,511 × 122), `application_test.csv` (48,744 × 121)
- **Note:** Data files are not committed (gitignored). Download from Kaggle and place under `data/`.

## Repo Layout

```
.
├── notebooks/                       # Jupyter notebooks (the project notebook = report)
│   ├── final_project_minwoo.ipynb   # neural-network track (this notebook)
│   ├── pmlm_utilities_shallow.py    # course utility (mirrored from companion)
│   └── _build_notebook.py           # script that regenerates the notebook
├── Nathaniel Badalov/                          # classic-ML companion notebook + its CV results
├── result/home_credit/              # CV CSVs, charts, Kaggle submission
├── slides/                          # final_presentation.pptx + builder + chart script
├── recording/script.md              # 8–10 min recording script
├── data/                            # Kaggle CSVs (gitignored)
└── README.md
```

## Reproduce

```bash
pip install scikit-learn lightgbm imbalanced-learn matplotlib seaborn pandas numpy
# place application_train.csv and application_test.csv under data/
jupyter notebook notebooks/final_project_minwoo.ipynb
```

To regenerate the slide deck and charts after re-running the notebook:

```bash
python slides/_build_charts.py
python slides/_build_slides.py
```
