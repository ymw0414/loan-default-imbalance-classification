"""
Builder script that constructs the final project notebook.
Run this once: `python _build_notebook.py` -> creates final_project_minwoo.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# =====================================================================
# 1. Title & Introduction
# =====================================================================
md("""# Loan Default Prediction with Neural Networks: A Class Imbalance Study

Companion notebook focusing on **neural network models** (shallow & deep MLP) for
the Home Credit Default Risk problem. Pipeline is intentionally aligned with the
classic-ML companion notebook (Logistic Regression / Random Forest / Histogram
Gradient Boosting) so that results across the two notebooks are directly comparable.""")

md("""## Introduction

This project predicts the probability that a loan applicant will default
(`TARGET = 1`) using the publicly available Kaggle *Home Credit Default Risk*
dataset. The training set contains 307,511 applications with 122 features and
is heavily **imbalanced** -- only ~8% of applicants default.

### Research question
Do neural networks respond differently to class imbalance handling than the
classic shallow models (LR, RF, HGBC)?

### Scope of this notebook
1. Mirror the preprocessing pipeline of the classic-ML notebook (same split,
   same imputation, same encoding, same scaling).
2. Train **Shallow MLP** and **Deep MLP** with `GridSearchCV` over a small
   hyperparameter grid using the same `PredefinedSplit` cross-validator.
3. **Bonus** -- on the best MLP architecture, compare three imbalance handling
   strategies: *baseline*, *SMOTE oversampling*, *threshold tuning*.
4. Generate Kaggle submission and combined comparison table.""")

# =====================================================================
# 2. Notebook Configuration
# =====================================================================
md("# Notebook Configuration")

md("## Warnings")
code("""import warnings
warnings.filterwarnings('ignore')""")

md("## Matplotlib")
code("""import matplotlib.pyplot as plt
%matplotlib inline""")

md("## Random seed")
code("""random_seed = 42""")

md("""## Paths

The notebook expects to be run from `final_project/notebooks/`. Data lives in
`../data/`. Result CSVs and submission file mirror Nathan's directory layout
(`../result/home_credit/...`).""")

code("""import os

# Absolute path to the project root (parent of notebooks/)
abspath_curr = os.path.abspath(os.path.join(os.getcwd(), '..')) + '/'
print('Project root:', abspath_curr)

# Path to data
data_dir = abspath_curr + 'data/'
print('Data dir:', data_dir)

# Path where results will be written
result_dir = abspath_curr + 'result/home_credit/'
os.makedirs(result_dir + 'cv_results/GridSearchCV/', exist_ok=True)
os.makedirs(result_dir + 'submission/', exist_ok=True)
os.makedirs(result_dir + 'figure/', exist_ok=True)
print('Result dir:', result_dir)""")

md("""## Course utility library

Imports the helper functions provided in `pmlm_utilities_shallow.py` (instructor's
shallow-learning utilities). The file lives next to this notebook.""")

code("""from pmlm_utilities_shallow import (
    common_var_checker,
    id_checker,
    nan_checker,
    cat_var_checker,
    get_train_val_ps,
)
import numpy as np
import pandas as pd""")

# =====================================================================
# 3. Data Preprocessing
# =====================================================================
md("# Data Preprocessing")

md("## Loading the data")
code("""df_raw_train = pd.read_csv(data_dir + 'application_train.csv', header=0)
df_train = df_raw_train.copy(deep=True)

df_raw_test = pd.read_csv(data_dir + 'application_test.csv', header=0)
df_test = df_raw_test.copy(deep=True)

target = 'TARGET'

print('train shape:', df_train.shape)
print('test  shape:', df_test.shape)
df_train.head()""")

md("""## Splitting the data (80% train / 20% validation)

Stratified split on `TARGET` to preserve the ~8% positive rate in both halves
(otherwise the class distribution can drift in a small validation slice).""")

code("""from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(
    df_train,
    train_size=0.8,
    random_state=random_seed,
    stratify=df_train[target],
)

df_train = df_train.reset_index(drop=True)
df_val   = df_val.reset_index(drop=True)

print('train:', df_train.shape, '  positive rate:', df_train[target].mean().round(4))
print('val  :', df_val.shape,   '  positive rate:', df_val[target].mean().round(4))""")

md("## Handling uncommon features")
code("""df_common_var = common_var_checker(df_train, df_val, df_test, target)
print('common variables:', len(df_common_var))

# Features in train that are not in val+test (and vice versa)
uncommon_train = np.setdiff1d(df_train.columns, df_common_var['common var'])
uncommon_val   = np.setdiff1d(df_val.columns,   df_common_var['common var'])
uncommon_test  = np.setdiff1d(df_test.columns,  df_common_var['common var'])
print('uncommon (train):', uncommon_train)
print('uncommon (val):  ', uncommon_val)
print('uncommon (test): ', uncommon_test)

if len(uncommon_train) > 0: df_train = df_train.drop(columns=uncommon_train)
if len(uncommon_val)   > 0: df_val   = df_val.drop(columns=uncommon_val)
if len(uncommon_test)  > 0: df_test  = df_test.drop(columns=uncommon_test)""")

md("## Handling identifiers (`SK_ID_CURR`)")
code("""df = pd.concat([df_train, df_val, df_test], sort=False)
df_id = id_checker(df)  # returns the *DataFrame* of identifier columns, not a list
id_cols = list(df_id.columns)
print('identifier columns:', id_cols)

# Save SK_ID_CURR for the submission file later
test_ids = df_test['SK_ID_CURR'].copy()

# Drop ID columns from all three frames
for c in id_cols:
    if c in df_train.columns: df_train = df_train.drop(columns=[c])
    if c in df_val.columns:   df_val   = df_val.drop(columns=[c])
    if c in df_test.columns:  df_test  = df_test.drop(columns=[c])

print('train shape after id drop:', df_train.shape)""")

md("""## Handling missing values

Mean imputation for **numeric** columns only. Categorical NaN is left as-is at
this point -- when `pd.get_dummies` encodes the categoricals later, rows with
NaN end up with zeros in all dummy columns of that variable, which is a
reasonable default category.""")

code("""df = pd.concat([df_train, df_val, df_test], sort=False)
df_nan = nan_checker(df)
print('# columns with NaN (any dtype):', len(df_nan))

# Keep only the numeric (float64) NaN columns -- categorical NaN is handled by get_dummies
df_miss = df_nan[df_nan['dtype'] == 'float64'].reset_index(drop=True)
print('# numeric columns with NaN:    ', len(df_miss))""")

code("""# Separate frames back
n_train_, n_val_ = df_train.shape[0], df_val.shape[0]
df_train = df.iloc[:n_train_, :].copy()
df_val   = df.iloc[n_train_:n_train_ + n_val_, :].copy()
df_test  = df.iloc[n_train_ + n_val_:, :].copy()""")

code("""from sklearn.impute import SimpleImputer

if len(df_miss['var']) > 0:
    si = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_train[df_miss['var']] = si.fit_transform(df_train[df_miss['var']])
    df_val[df_miss['var']]   = si.transform(df_val[df_miss['var']])
    df_test[df_miss['var']]  = si.transform(df_test[df_miss['var']])

# Numeric residuals should now be zero; categorical NaN remains until get_dummies.
print('residual numeric NaN in train:', df_train.select_dtypes(include=[np.number]).isna().sum().sum())""")

md("""## Encoding the categorical features

One-hot encoding via `pd.get_dummies`. Combining the three frames before
encoding guarantees that train/val/test end up with the same set of dummy
columns even if a particular category only appears in one of them.

> Note: pandas 3.0 stores string columns with dtype `'str'` instead of
> `'object'`, so we pull categorical columns with `select_dtypes` rather than
> the course utility's default `dtype='object'` filter.""")

code("""df = pd.concat([df_train, df_val, df_test], sort=False)

# Pandas 3.0 may use 'str' or 'string'; older versions use 'object'.
cat_cols = df.select_dtypes(include=['object', 'str', 'string']).columns.tolist()
print('categorical columns:', cat_cols)

df = pd.get_dummies(df, columns=[c for c in cat_cols if c != target])
print('shape after get_dummies:', df.shape)
df.head()""")

md("## Splitting back into train / val / test and pulling out X, y")
code("""# split back
df_train = df.iloc[:df_train.shape[0], :].copy()
df_val   = df.iloc[df_train.shape[0]:df_train.shape[0] + df_val.shape[0], :].copy()
df_test  = df.iloc[df_train.shape[0] + df_val.shape[0]:, :].copy()

# X / y
feature_cols = np.setdiff1d(df_train.columns, [target])

X_train = df_train[feature_cols].values
X_val   = df_val[feature_cols].values
X_test  = df_test[feature_cols].values

y_train = df_train[target].values.astype(int)
y_val   = df_val[target].values.astype(int)
# y_test does not exist (Kaggle hidden labels)

print('X_train:', X_train.shape, '  y_train positive rate:', y_train.mean().round(4))
print('X_val:  ', X_val.shape,   '  y_val   positive rate:', y_val.mean().round(4))
print('X_test: ', X_test.shape)""")

md("""## Scaling the features

`MLPClassifier` is **highly sensitive to feature scale** -- without
standardization, training is slow and unstable. Same `StandardScaler` instance
is fitted on the training set and applied to val and test.""")

code("""from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_val   = ss.transform(X_val)
X_test  = ss.transform(X_test)

print('after scaling -- mean:', X_train.mean().round(4), '  std:', X_train.std().round(4))""")

# =====================================================================
# 4. Hyperparameter Tuning - Neural Networks
# =====================================================================
md("""# Hyperparameter Tuning -- Neural Networks

Two architectures, both implemented with `sklearn.neural_network.MLPClassifier`.

| Model | Architecture | Course slide |
|---|---|---|
| **Shallow MLP** | `(64,)` -- one hidden layer | 3/16 *Shallow Neural Networks* |
| **Deep MLP**    | `(128, 64, 32)` -- three hidden layers | 3/30 *Deep Neural Networks* |

Both use ReLU activations, Adam optimizer, and `early_stopping=True` so
training automatically halts when the validation loss stops improving.""")

md("## Creating the dictionary of models")
code("""from sklearn.neural_network import MLPClassifier

models = {
    'shallow_mlp': MLPClassifier(
        hidden_layer_sizes=(64,),
        activation='relu',
        solver='adam',
        early_stopping=True,
        max_iter=100,
        random_state=random_seed,
    ),
    'deep_mlp': MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        early_stopping=True,
        max_iter=100,
        random_state=random_seed,
    ),
}""")

md("## Wrapping the models in `Pipeline`")
code("""from sklearn.pipeline import Pipeline

pipes = {acronym: Pipeline([('model', m)]) for acronym, m in models.items()}""")

md("## Predefined split cross-validator")
code("""# Combines train+val into one array; PredefinedSplit tells GridSearchCV
# to use the val portion as the validation fold (no random k-fold).
X_train_val, y_train_val, ps = get_train_val_ps(X_train, y_train, X_val, y_val)
print('combined train+val:', X_train_val.shape)""")

md("""## Hyperparameter grids

Kept small on purpose -- each MLP fit on ~245k rows with ~200 features is not
cheap, and we have a one-day deadline.""")

code("""param_grids = {}

# Shallow MLP
param_grids['shallow_mlp'] = [{
    'model__alpha':              [1e-4, 1e-3],
    'model__learning_rate_init': [0.001, 0.005],
}]

# Deep MLP
param_grids['deep_mlp'] = [{
    'model__alpha':              [1e-4, 1e-3],
    'model__learning_rate_init': [0.001, 0.005],
}]""")

md("## Running `GridSearchCV`")
code("""from sklearn.model_selection import GridSearchCV

best_score_params_estimator_gs = []  # one entry per model: [score, params, estimator]

for acronym in pipes.keys():
    print(f'\\n>>> tuning {acronym} ...')
    gs = GridSearchCV(
        estimator=pipes[acronym],
        param_grid=param_grids[acronym],
        scoring='roc_auc',
        n_jobs=2,
        cv=ps,
        return_train_score=True,
        verbose=1,
    )
    gs = gs.fit(X_train_val, y_train_val)

    best_score_params_estimator_gs.append([
        gs.best_score_,
        gs.best_params_,
        gs.best_estimator_,
    ])

    cv_results = pd.DataFrame.from_dict(gs.cv_results_).sort_values(
        by=['rank_test_score', 'std_test_score']
    )
    important_cols = [
        'rank_test_score', 'mean_test_score', 'std_test_score',
        'mean_train_score', 'std_train_score',
        'mean_fit_time', 'std_fit_time',
        'mean_score_time', 'std_score_time',
    ]
    cv_results = cv_results[
        important_cols + sorted(list(set(cv_results.columns) - set(important_cols)))
    ]
    cv_results.to_csv(
        result_dir + f'cv_results/GridSearchCV/{acronym}.csv',
        index=False,
    )

    print(f'    best ROC AUC = {gs.best_score_:.4f}')
    print(f'    best params  = {gs.best_params_}')""")

# =====================================================================
# 5. Model Selection
# =====================================================================
md("# Model Selection")
code("""best_score_params_estimator_gs = sorted(
    best_score_params_estimator_gs, key=lambda x: x[0], reverse=True
)

results_table = pd.DataFrame(
    [[s, p] for s, p, _ in best_score_params_estimator_gs],
    columns=['best_val_AUC', 'best_params'],
    index=['rank_' + str(i+1) for i in range(len(best_score_params_estimator_gs))],
)
results_table""")

# =====================================================================
# 6. Imbalance Handling Comparison (Bonus)
# =====================================================================
md("""# Imbalance Handling Comparison (Bonus)

`MLPClassifier` does **not** support `class_weight`, so we compare three
techniques that *do* work for it:

1. **Baseline** -- the best MLP trained on the original (imbalanced) data.
2. **SMOTE** -- synthetic minority oversampling so the training set is 50/50.
3. **Threshold tuning** -- train as in (1), but pick the decision threshold
   that maximizes F1 on the validation set instead of the default 0.5.

For all three, ROC AUC, PR AUC, and recall/precision/F1 at the chosen
threshold are reported on the validation set.""")

md("## Picking the best MLP architecture")
code("""# Use the higher-AUC architecture from the GridSearchCV run above.
best_score, best_params, best_estimator = best_score_params_estimator_gs[0]
print('Best NN architecture from grid search:')
print('  AUC   :', round(best_score, 4))
print('  params:', best_params)

# Identify which acronym this corresponds to (shallow vs deep)
best_acronym = None
for acronym, p in pipes.items():
    if type(p.named_steps['model']) is type(best_estimator.named_steps['model']):
        if p.named_steps['model'].hidden_layer_sizes == best_estimator.named_steps['model'].hidden_layer_sizes:
            best_acronym = acronym
            break
print('  -> picking architecture:', best_acronym)""")

md("## Helper: evaluation metrics")
code("""from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix,
)

def evaluate(y_true, y_proba, threshold=0.5, label=''):
    y_pred = (y_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_proba)
    pr  = average_precision_score(y_true, y_proba)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {
        'method': label,
        'threshold': round(threshold, 4),
        'ROC_AUC': round(auc, 4),
        'PR_AUC':  round(pr, 4),
        'precision': round(p, 4),
        'recall':    round(r, 4),
        'F1':        round(f1, 4),
    }""")

md("## (1) Baseline")
code("""baseline_proba = best_estimator.predict_proba(X_val)[:, 1]
res_baseline = evaluate(y_val, baseline_proba, threshold=0.5, label='baseline')
res_baseline""")

md("""## (2) SMOTE oversampling

`imblearn.over_sampling.SMOTE` synthesizes new minority-class samples by
interpolating between existing minority points. We apply it **only to the
training data** (never to validation -- that would leak information).""")

code("""from imblearn.over_sampling import SMOTE

print('train class distribution before SMOTE:', np.bincount(y_train))

sm = SMOTE(random_state=random_seed)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print('train class distribution after  SMOTE:', np.bincount(y_train_sm))""")

code("""# Re-train MLP with the same architecture but on SMOTE-balanced data
mlp_smote = MLPClassifier(
    hidden_layer_sizes=best_estimator.named_steps['model'].hidden_layer_sizes,
    activation='relu',
    solver='adam',
    alpha=best_estimator.named_steps['model'].alpha,
    learning_rate_init=best_estimator.named_steps['model'].learning_rate_init,
    early_stopping=True,
    max_iter=100,
    random_state=random_seed,
)
mlp_smote.fit(X_train_sm, y_train_sm)
smote_proba = mlp_smote.predict_proba(X_val)[:, 1]
res_smote = evaluate(y_val, smote_proba, threshold=0.5, label='SMOTE')
res_smote""")

md("""## (3) Threshold tuning

The baseline MLP already gives us probabilities. Instead of using the default
threshold of 0.5, we sweep a range of thresholds on the **validation set** and
pick the one that maximizes F1.""")

code("""from sklearn.metrics import precision_recall_curve

# Use the baseline model's predicted probabilities
prec, rec, thr = precision_recall_curve(y_val, baseline_proba)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = int(np.nanargmax(f1s))
best_thr = float(thr[best_idx]) if best_idx < len(thr) else 0.5
print('best threshold (max F1 on val):', round(best_thr, 4))

res_threshold = evaluate(y_val, baseline_proba, threshold=best_thr, label='threshold')
res_threshold""")

md("## Comparison table")
code("""imbalance_results = pd.DataFrame([res_baseline, res_smote, res_threshold])
imbalance_results.to_csv(result_dir + 'cv_results/GridSearchCV/imbalance_compare.csv', index=False)
imbalance_results""")

md("""### Quick interpretation

* If **baseline** already has the highest ROC AUC, the MLP is robust enough that
  imbalance handling provides no AUC gain -- the gain (if any) shows up as a
  recall increase from threshold tuning.
* If **SMOTE** improves ROC AUC, the model was indeed under-using the minority
  class.
* **Threshold tuning** never changes ROC AUC (it's a post-hoc decision rule),
  but typically trades precision for recall.""")

# =====================================================================
# 7. Combined Results (Nathan + Minwoo)
# =====================================================================
md("""# Combined Results: Classic ML + Neural Networks

Loads Nathan's `GridSearchCV` CSVs (LR / RF / HGBC) and merges them with the
two MLP results from this notebook. This gives the full **5-model comparison**
that we present in the final slides.""")

code("""def best_row(path):
    df = pd.read_csv(path)
    df = df.sort_values('rank_test_score').head(1)
    return df['mean_test_score'].iloc[0], df['params'].iloc[0]

cv_path = result_dir + 'cv_results/GridSearchCV/'

rows = []
for fname, model_label in [
    ('lr.csv',           'Logistic Regression (Nathan)'),
    ('rfc.csv',          'Random Forest (Nathan)'),
    ('hgbc.csv',         'HistGradBoost (Nathan)'),
    ('shallow_mlp.csv',  'Shallow MLP (Minwoo)'),
    ('deep_mlp.csv',     'Deep MLP (Minwoo)'),
]:
    p = cv_path + fname
    if os.path.exists(p):
        score, params = best_row(p)
        rows.append({'model': model_label, 'best_val_AUC': round(score, 4), 'best_params': params})
    else:
        rows.append({'model': model_label, 'best_val_AUC': None, 'best_params': '<missing CSV>'})

combined = pd.DataFrame(rows).sort_values('best_val_AUC', ascending=False, na_position='last').reset_index(drop=True)
combined.to_csv(result_dir + 'combined_model_comparison.csv', index=False)
combined""")

# =====================================================================
# 8. Submission Generation
# =====================================================================
md("""# Submission Generation

Train the overall best NN on the **full training data** (train + val
concatenated, before SMOTE) and predict probabilities on the held-out Kaggle
test set.""")

code("""# Best NN architecture, retrained on the full train+val data
best_mlp = MLPClassifier(
    hidden_layer_sizes=best_estimator.named_steps['model'].hidden_layer_sizes,
    activation='relu',
    solver='adam',
    alpha=best_estimator.named_steps['model'].alpha,
    learning_rate_init=best_estimator.named_steps['model'].learning_rate_init,
    early_stopping=True,
    max_iter=100,
    random_state=random_seed,
)
best_mlp.fit(X_train_val, y_train_val)

test_proba = best_mlp.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({'SK_ID_CURR': test_ids.values, 'TARGET': test_proba})
out_path = result_dir + 'submission/submission_minwoo_mlp.csv'
submission.to_csv(out_path, index=False)
print('written:', out_path)
submission.head()""")

# =====================================================================
# 9. Interpretation & Conclusion
# =====================================================================
md("""# Interpretation

1. **Among neural networks**, the [shallow / deep] MLP achieves higher
   validation ROC AUC. Update this paragraph after running the cells above.

2. **Compared to classic ML** (Nathan), the best NN ranks [#] in the combined
   table. Tabular credit-default data tends to favor gradient-boosted trees,
   so a slightly lower NN score here is consistent with the literature.

3. **Imbalance handling**:
   * Baseline ROC AUC = ?
   * SMOTE ROC AUC    = ?
   * Threshold tuning trades ~? precision for ~? recall at the F1-optimal cutoff.

   Replace the `?`s after running the imbalance comparison cell.""")

md("""# Conclusion

* **Best model overall**: ... (filled in after the run).
* **Effect of imbalance handling on the MLP**: ...
* **Limitations**: only the main `application_*.csv` tables are used; the
  bureau / previous_application / installments tables would likely close most
  of the gap to the public-leaderboard top scores.
* **Future work**: integrate the auxiliary tables; ensemble the best NN with
  HGBC; calibrate probabilities (Platt scaling) before submission.""")

# =====================================================================
# Write out
# =====================================================================
nb.cells = cells
nb.metadata = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language':     'python',
        'name':         'python3',
    },
    'language_info': {
        'name':            'python',
        'pygments_lexer':  'ipython3',
    },
}

out_file = 'final_project_minwoo.ipynb'
with open(out_file, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'wrote {out_file} with {len(cells)} cells')
