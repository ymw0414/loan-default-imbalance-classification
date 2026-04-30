# Data

The Home Credit Default Risk CSVs are not committed to this repository — they
total ~700 MB and are gitignored. Download them from Kaggle before running the
notebook.

## Required files

Place these under `data/` (this directory):

- `application_train.csv` — 307,511 × 122 (training set with `TARGET`)
- `application_test.csv` — 48,744 × 121 (test set, no `TARGET`)
- `sample_submission.csv` — Kaggle submission template
- `HomeCredit_columns_description.csv` — column documentation (optional)

Auxiliary tables (`bureau.csv`, `previous_application.csv`, etc.) are not used.

## Download

```bash
# via Kaggle CLI
kaggle competitions download -c home-credit-default-risk -p data/
unzip data/home-credit-default-risk.zip -d data/
```

Or download manually from
<https://www.kaggle.com/competitions/home-credit-default-risk/data>.
