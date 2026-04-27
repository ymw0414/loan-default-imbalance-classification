"""
Builder script that constructs the final project notebook.
Run this once: `python _build_notebook.py` -> creates final_project.ipynb
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
md("""# Loan Default Prediction: Comparing Classic ML and Neural Networks under Class Imbalance

**Team:** Minwoo Yoo · Nathaniel Badalov
**Dataset:** Home Credit Default Risk (Kaggle)

This notebook is the **combined final report**. It walks through the full
pipeline -- shared preprocessing, classic shallow learners (Logistic
Regression, Random Forest, Histogram Gradient Boosting), shallow & deep
neural networks, and a focused study of how each family responds to class
imbalance handling.

**Authorship of each section is marked inline.** The classic-ML hyperparameter
tuning was driven by Nathaniel Badalov; the neural-network hyperparameter tuning and
the imbalance-handling deep dive were driven by Minwoo Yoo. EDA, preprocessing,
combined results and the conclusion were written jointly.""")

md("""## Introduction

### Business problem
Consumer lenders need to estimate, before granting a loan, the probability
that the applicant will fail to repay it. Misjudging this probability is
asymmetric: approving a borrower who later defaults costs the full unpaid
principal, while denying a creditworthy borrower only forfeits the interest
margin. Models that **rank** applicants well let lenders set a sensible
approval threshold and trade off these two costs explicitly.

### Dataset
We use the publicly available Kaggle *Home Credit Default Risk* dataset.
The training file contains 307,511 applications with 122 features --
demographics (age, gender, family status), employment info, credit-bureau
summaries, the requested loan amount, and several pre-computed external
credit scores (`EXT_SOURCE_1/2/3`). The target is binary:

  * `TARGET = 1` -- the applicant defaulted (24,825 rows, **~8.07%**)
  * `TARGET = 0` -- the applicant repaid normally (282,686 rows, ~91.93%)

The held-out Kaggle test set has 48,744 applicants whose labels are hidden.

### Why class imbalance matters here
With only ~8% positives, the **default decision rule** of "predict 1 if
P(default) ≥ 0.5" tends to break: a model can land at 0.745 ROC AUC -- so it
*ranks* applicants correctly -- yet still classify zero applicants as
defaulters because none of its predicted probabilities make it over 0.5.
That means a useful ranker can produce a useless classifier if we ignore
imbalance. Three families of fixes are widely taught:

  * **Reweighting the loss** (`class_weight='balanced'`) -- charge the
    minority class more for each mistake.
  * **Resampling the training set** (e.g. SMOTE) -- synthesize minority
    samples until the training distribution is balanced.
  * **Tuning the decision threshold** post-hoc -- keep the trained model,
    just stop using 0.5 as the cutoff.

### Research question
> Do neural networks respond to class imbalance handling techniques the
> same way classic shallow learners (Logistic Regression, Random Forest,
> Histogram Gradient Boosting) do?

### Scope of this notebook
1. Reproduce the preprocessing pipeline used for the classic-ML companion
   notebook (same split, same imputation, same encoding, same scaling),
   so that NN results are directly comparable.
2. Tune **Shallow MLP** and **Deep MLP** with `GridSearchCV`, using the same
   `PredefinedSplit` cross-validator.
3. On the best MLP, compare three imbalance handling strategies --
   *baseline*, *SMOTE oversampling*, *threshold tuning*.
4. Combine results with the classic-ML CSVs into a single 5-model
   leaderboard, generate a Kaggle submission, and discuss findings.""")

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
`../data/`. Result CSVs and submission file mirror Nathaniel Badalov's directory layout
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

md("""## Exploratory Data Analysis

A short look at the data **before any preprocessing** so the design choices
that follow are transparent. We confirm the class imbalance, look at the
strongest predictors, and quantify the missing-value problem.""")

md("### Target distribution")
code("""tgt_counts = df_train[target].value_counts().sort_index()
tgt_pct    = (tgt_counts / tgt_counts.sum() * 100).round(2)
display(pd.DataFrame({'count': tgt_counts, 'percent': tgt_pct}))

fig, ax = plt.subplots(1, 1, figsize=(7, 3))
colors = ['#2E86C1', '#E74C3C']
ax.barh(['No default (0)', 'Default (1)'], tgt_counts.values, color=colors)
for i, (c, p) in enumerate(zip(tgt_counts.values, tgt_pct.values)):
    ax.text(c + 4000, i, f'{c:,}  ({p:.2f}%)', va='center')
ax.set_xlim(0, tgt_counts.max() * 1.18)
ax.set_xlabel('# applicants')
ax.set_title('Severe class imbalance — only 8.07% defaults')
for s in ('top', 'right'): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.show()""")

md("""### Strongest numeric predictors

Pearson correlation of every numeric column with `TARGET`. The three
external credit scores (`EXT_SOURCE_*`) dominate the ranking by a wide
margin -- they are the most informative single features in the file.""")

code("""num_cols = df_train.select_dtypes(include='number').columns.tolist()
corr = df_train[num_cols].corr()[target].drop(target)

top_pos = corr.sort_values(ascending=False).head(8)
top_neg = corr.sort_values(ascending=True).head(8)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].barh(top_neg.index[::-1], top_neg.values[::-1], color='#2E86C1')
axes[0].set_title('Top 8 negative correlations  (lower default risk →)')
axes[0].axvline(0, color='black', lw=0.5)
for s in ('top', 'right'): axes[0].spines[s].set_visible(False)

axes[1].barh(top_pos.index[::-1], top_pos.values[::-1], color='#E74C3C')
axes[1].set_title('Top 8 positive correlations  (higher default risk →)')
axes[1].axvline(0, color='black', lw=0.5)
for s in ('top', 'right'): axes[1].spines[s].set_visible(False)

plt.tight_layout(); plt.show()""")

md("""### Missing-value landscape

Many columns are missing in 50–70% of rows -- the *building information*
columns (`APARTMENTS_AVG`, `LIVINGAREA_*`, ...) and `OWN_CAR_AGE`. We will
mean-impute the numeric ones and let `pd.get_dummies` produce all-zero
indicator blocks for categorical NaNs.""")

code("""miss = df_train.isna().mean().sort_values(ascending=False)
miss = miss[miss > 0]
print(f'columns with at least one NaN: {len(miss)} / {df_train.shape[1] - 1}')
print(f'columns with >= 50% NaN:       {(miss >= 0.5).sum()}')

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(miss)), miss.values * 100, color='#7FB3D5')
ax.axhline(50, color='#E74C3C', lw=0.7, linestyle='--', label='50% threshold')
ax.set_xlabel('columns (sorted by missing rate, descending)')
ax.set_ylabel('% missing')
ax.set_title('Missing rate per column')
ax.legend()
for s in ('top', 'right'): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.show()""")

md("""### Categorical columns and cardinality

We have 16 categorical (string) columns. Most are low-cardinality
(2–8 categories), but `ORGANIZATION_TYPE` has 58 categories -- aggressive
one-hot encoding would create a wide sparse block, which tends to hurt
neural networks more than tree models.""")

code("""cat_view = df_train.select_dtypes(include=['object', 'str', 'string'])
card = cat_view.nunique().sort_values(ascending=False)
display(pd.DataFrame({'n_unique': card}))""")

md("""### EDA takeaways

* **Imbalanced target** (~8% positives) → choose ROC AUC over accuracy and
  expect to need a custom decision threshold.
* **EXT_SOURCE_*** features carry most of the signal; preserving them
  (mean-imputing rather than dropping) is essential.
* **Heavy missingness** in building-info columns is structural (the lender
  simply didn't have those fields) -- the missing-indicator pattern itself
  may carry signal, which one-hot dummies will partly capture.
* **`ORGANIZATION_TYPE` is the only high-cardinality categorical** -- a
  tree-based learner can split on its dummies cheaply, but the MLP has to
  weight ~58 sparse columns simultaneously.""")

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
# 4. Hyperparameter Tuning -- Classic ML  (Nathaniel Badalov)
# =====================================================================
md("""# Hyperparameter Tuning -- Classic ML

> **Author:** Nathaniel Badalov
>
> Three classic shallow learners compared: Logistic Regression (linear),
> Random Forest (tree bagging), and HistGradientBoosting (tree boosting).
> All wrapped in the same `Pipeline` and tuned with the same `GridSearchCV`
> over the predefined train/val split, scoring on ROC AUC.""")

md("## Models")
code("""from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone   # used by the NN smart-skip block below

classic_models = {
    'lr':   LogisticRegression(class_weight='balanced', random_state=random_seed, max_iter=200),
    'rfc':  RandomForestClassifier(class_weight='balanced', random_state=random_seed),
    'hgbc': HistGradientBoostingClassifier(random_state=random_seed),
}

classic_pipes = {acronym: Pipeline([('model', m)]) for acronym, m in classic_models.items()}""")

md("## Predefined train/val split (shared with the NN section)")
code("""# Combines train+val into one array; PredefinedSplit tells GridSearchCV
# to use the val portion as the validation fold (no random k-fold).
X_train_val, y_train_val, ps = get_train_val_ps(X_train, y_train, X_val, y_val)
print('combined train+val:', X_train_val.shape)""")

md("""## Hyperparameter grids

Same grids Nathaniel Badalov used in his standalone notebook. Random Forest is the slowest
of the three (full grid takes ~10 minutes); a smart-skip block below loads
existing CSVs from disk if they're already there to avoid re-running.""")

code("""classic_param_grids = {
    'lr': [{
        'model__tol': [1e-5, 1e-4, 1e-3],
        'model__C':   [0.01, 0.1, 1, 10],
    }],
    'rfc': [{
        'model__n_estimators':      [100, 200],
        'model__min_samples_split': [2, 20, 100],
        'model__min_samples_leaf':  [1, 20, 100],
    }],
    'hgbc': [{
        'model__learning_rate':    [0.01, 0.05, 0.1],
        'model__max_iter':         [100, 200],
        'model__min_samples_leaf': [20, 100],
    }],
}""")

md("""## Running `GridSearchCV` on the classic models

If the result CSV for a model already exists on disk we *load it* instead of
retraining. This makes re-runs fast (the heavy lifting only happens once).
Delete the CSV files in `result/home_credit/cv_results/GridSearchCV/` to force
a full retrain.""")

code("""from sklearn.model_selection import GridSearchCV

classic_results = []   # [best_score, best_params_dict, best_estimator_or_None]
cv_dir = result_dir + 'cv_results/GridSearchCV/'

for acronym in classic_pipes.keys():
    csv_path = cv_dir + acronym + '.csv'
    if os.path.exists(csv_path):
        # ---- skip retraining: load best-row from existing CSV ----
        df_cv = pd.read_csv(csv_path).sort_values('rank_test_score').head(1)
        best_score  = float(df_cv['mean_test_score'].iloc[0])
        # 'params' column is a stringified dict; eval is acceptable here because
        # it was written by us in the same notebook
        import ast
        best_params = ast.literal_eval(df_cv['params'].iloc[0])
        classic_results.append([best_score, best_params, None])
        print(f'{acronym}: loaded from {csv_path}  (best ROC AUC = {best_score:.4f})')
    else:
        print(f'\\n>>> tuning {acronym} ...')
        gs = GridSearchCV(
            estimator=classic_pipes[acronym],
            param_grid=classic_param_grids[acronym],
            scoring='roc_auc',
            n_jobs=2,
            cv=ps,
            return_train_score=True,
            verbose=1,
        )
        gs = gs.fit(X_train_val, y_train_val)
        classic_results.append([gs.best_score_, gs.best_params_, gs.best_estimator_])

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
        cv_results.to_csv(csv_path, index=False)
        print(f'    best ROC AUC = {gs.best_score_:.4f}')
        print(f'    best params  = {gs.best_params_}')""")

md("## Classic ML leaderboard")
code("""classic_leaderboard = pd.DataFrame(
    [{'model': name, 'best_val_AUC': round(s, 4), 'best_params': p}
     for name, (s, p, _) in zip(['lr', 'rfc', 'hgbc'], classic_results)]
).sort_values('best_val_AUC', ascending=False).reset_index(drop=True)
classic_leaderboard""")

# =====================================================================
# 5. Hyperparameter Tuning - Neural Networks  (Minwoo Yoo)
# =====================================================================
md("""# Hyperparameter Tuning -- Neural Networks

> **Author:** Minwoo Yoo
>
> Two architectures, both implemented with `sklearn.neural_network.MLPClassifier`.

| Model | Architecture | Course slide |
|---|---|---|
| **Shallow MLP** | `(64,)` -- one hidden layer | 3/16 *Shallow Neural Networks* |
| **Deep MLP**    | `(128, 64, 32)` -- three hidden layers | 3/30 *Deep Neural Networks* |

Both use ReLU activations, Adam optimizer, and `early_stopping=True` so
training automatically halts when the validation loss stops improving.""")

md("## Creating the dictionary of models")
code("""from sklearn.neural_network import MLPClassifier

nn_models = {
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
}

nn_pipes = {acronym: Pipeline([('model', m)]) for acronym, m in nn_models.items()}""")

md("""## Hyperparameter grids

Kept small on purpose -- each MLP fit on ~245k rows with ~200 features is not
cheap, and we have a one-day deadline.""")

code("""nn_param_grids = {
    'shallow_mlp': [{
        'model__alpha':              [1e-4, 1e-3],
        'model__learning_rate_init': [0.001, 0.005],
    }],
    'deep_mlp': [{
        'model__alpha':              [1e-4, 1e-3],
        'model__learning_rate_init': [0.001, 0.005],
    }],
}""")

md("""## Running `GridSearchCV` on the NN models

Same smart-skip pattern: if the model's CV CSV already exists on disk, we load
the best row instead of retraining.""")

code("""nn_results = []

for acronym in nn_pipes.keys():
    csv_path = cv_dir + acronym + '.csv'
    if os.path.exists(csv_path):
        df_cv = pd.read_csv(csv_path).sort_values('rank_test_score').head(1)
        best_score  = float(df_cv['mean_test_score'].iloc[0])
        import ast
        best_params = ast.literal_eval(df_cv['params'].iloc[0])
        # Rebuild a fitted estimator with the best params so downstream sections
        # (imbalance dive, submission) can use it without retraining the grid.
        best_pipe = clone(nn_pipes[acronym])
        best_pipe.set_params(**best_params)
        best_pipe.fit(X_train_val, y_train_val)
        nn_results.append([best_score, best_params, best_pipe])
        print(f'{acronym}: loaded from {csv_path}  (best ROC AUC = {best_score:.4f})')
    else:
        print(f'\\n>>> tuning {acronym} ...')
        gs = GridSearchCV(
            estimator=nn_pipes[acronym],
            param_grid=nn_param_grids[acronym],
            scoring='roc_auc',
            n_jobs=2,
            cv=ps,
            return_train_score=True,
            verbose=1,
        )
        gs = gs.fit(X_train_val, y_train_val)
        nn_results.append([gs.best_score_, gs.best_params_, gs.best_estimator_])

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
        cv_results.to_csv(csv_path, index=False)
        print(f'    best ROC AUC = {gs.best_score_:.4f}')
        print(f'    best params  = {gs.best_params_}')""")

# =====================================================================
# 5. Model Selection (across all 5 models)
# =====================================================================
md("""# Model Selection -- All Five Models

Combines the classic-ML and neural-network results into one leaderboard.""")

code("""all_results = []
for name, (s, p, est) in zip(['Logistic Regression', 'Random Forest', 'HistGradientBoosting'],
                              classic_results):
    all_results.append({'model': name, 'family': 'classic',
                        'best_val_AUC': round(s, 4), 'best_params': p, 'estimator': est})
for name, (s, p, est) in zip(['Shallow MLP', 'Deep MLP'], nn_results):
    all_results.append({'model': name, 'family': 'neural net',
                        'best_val_AUC': round(s, 4), 'best_params': p, 'estimator': est})

leaderboard = pd.DataFrame(all_results).sort_values('best_val_AUC', ascending=False).reset_index(drop=True)
leaderboard[['model', 'family', 'best_val_AUC', 'best_params']]""")

md("""## Best-of-each-family

A useful sanity check: the strongest representative of each model family.""")

code("""leaderboard.groupby('family').apply(lambda g: g.nlargest(1, 'best_val_AUC'))[['model', 'best_val_AUC']]""")

# =====================================================================
# 6. Imbalance Handling Comparison (Bonus)
# =====================================================================
md("""# Imbalance Handling Comparison (Bonus)

The classic-ML companion notebook applies `class_weight='balanced'` to LR
and Random Forest as a default. `MLPClassifier`, however, does **not**
expose `class_weight`, so the cleanest way to study imbalance handling on
the neural-network side is to compare three sklearn-friendly strategies:

| # | Strategy | Acts on | Description |
|---|---|---|---|
| 1 | **Baseline** | nothing | Train the best MLP on the original imbalanced training set; classify at threshold 0.5. |
| 2 | **SMOTE** | training data | Synthesize minority-class samples by interpolating between existing minority points (k-nearest-neighbor in feature space) until the training set is 50/50, then re-train the MLP. |
| 3 | **Threshold tuning** | decision rule | Re-use the baseline model but pick the decision threshold that maximizes F1 on the validation set, instead of using the default 0.5. |

A short walk-through of each technique is given below before we evaluate
them. ROC AUC and PR AUC are reported on the validation set together with
precision/recall/F1 at each method's chosen threshold.

> **Why three rather than four?** `class_weight` is intentionally omitted
> here because sklearn's `MLPClassifier` does not implement it; the
> reweighting comparison instead lives in the classic-ML notebook for LR
> and Random Forest.""")

md("""## Picking the best MLP architecture (and re-fitting on X_train only)

> **Methodological note (per Nathaniel's review).**
> `GridSearchCV` returns its `best_estimator_` re-fit on the *full* data we
> passed in (here `X_train_val`, i.e. train $+$ val). If we then evaluated
> imbalance methods on `X_val`, the model would already have seen those
> rows during refit, inflating the comparison.
>
> We side-step this by **re-training a fresh MLP on `X_train` only**, using
> the best hyperparameters that GridSearchCV found. The fresh estimator is
> what we use for all imbalance experiments below.""")

code("""# Pick the higher-AUC of the two NN architectures (from the held-out grid search).
nn_sorted = sorted(nn_results, key=lambda x: x[0], reverse=True)
best_score, best_params, best_estimator_grid = nn_sorted[0]

print('Best NN from grid search (held-out val):')
print('  val ROC AUC:', round(best_score, 4))
print('  params     :', best_params)

# Identify shallow vs deep
arch = best_estimator_grid.named_steps['model'].hidden_layer_sizes
print('  hidden     :', arch,
      '(shallow)' if len(arch) == 1 else '(deep)')""")

code("""# Re-train a fresh MLP on X_train ONLY (avoids the GridSearchCV refit-on-train+val leak).
fresh_alpha = best_estimator_grid.named_steps['model'].alpha
fresh_lr    = best_estimator_grid.named_steps['model'].learning_rate_init

best_estimator = MLPClassifier(
    hidden_layer_sizes=arch,
    activation='relu',
    solver='adam',
    alpha=fresh_alpha,
    learning_rate_init=fresh_lr,
    early_stopping=True,
    max_iter=100,
    random_state=random_seed,
)
best_estimator.fit(X_train, y_train)
print('fresh MLP fitted on X_train only ({} rows)'.format(X_train.shape[0]))""")

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

md("""## (1) Baseline -- no imbalance handling

We use the best MLP from the grid search as is, with the default decision
rule of "predict 1 iff P(default) ≥ 0.5". This serves as the reference
point: anything more elaborate has to beat these numbers to be worth
keeping.""")
code("""baseline_proba = best_estimator.predict_proba(X_val)[:, 1]
res_baseline = evaluate(y_val, baseline_proba, threshold=0.5, label='baseline')
res_baseline""")

code("""# Diagnostic: how many validation samples cross the 0.5 cutoff?
n_pos = int((baseline_proba >= 0.5).sum())
print(f'predictions above 0.5: {n_pos} / {len(baseline_proba)}'
      f'  ({n_pos/len(baseline_proba)*100:.2f}%)')
print(f'predicted-probability range : [{baseline_proba.min():.4f}, {baseline_proba.max():.4f}]')
print(f'predicted-probability median: {np.median(baseline_proba):.4f}')""")

md("""## (2) SMOTE oversampling

**SMOTE** (Synthetic Minority Over-sampling TEchnique, Chawla et al. 2002)
balances the training set by *manufacturing* extra minority-class samples:

1. Pick a minority sample `x`.
2. Find its `k` nearest minority-class neighbors (`k=5` is the default).
3. Pick one neighbor `x'` at random.
4. Generate a new synthetic point at a random position along the line
   segment between `x` and `x'`.
5. Repeat until the minority class has the same size as the majority class.

We apply SMOTE **only to the training data** -- adding synthetic points to
the validation set would leak information and inflate the score. After
oversampling, the same MLP architecture is re-trained on the now-balanced
training set.""")

code("""from imblearn.over_sampling import SMOTE

print('train class distribution before SMOTE:', np.bincount(y_train))

sm = SMOTE(random_state=random_seed)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print('train class distribution after  SMOTE:', np.bincount(y_train_sm))""")

code("""# Re-train MLP with the same architecture but on SMOTE-balanced data.
# Uses the same hyperparameters as the fresh baseline so SMOTE vs baseline
# is a fair head-to-head (only the training distribution differs).
mlp_smote = MLPClassifier(
    hidden_layer_sizes=arch,
    activation='relu',
    solver='adam',
    alpha=fresh_alpha,
    learning_rate_init=fresh_lr,
    early_stopping=True,
    max_iter=100,
    random_state=random_seed,
)
mlp_smote.fit(X_train_sm, y_train_sm)
smote_proba = mlp_smote.predict_proba(X_val)[:, 1]
res_smote = evaluate(y_val, smote_proba, threshold=0.5, label='SMOTE')
res_smote""")

md("""## (3) Threshold tuning

Threshold tuning operates on the **decision rule**, not the model: we keep
the baseline MLP's predicted probabilities and ask "what cutoff gives the
best classifier?" Concretely, for every threshold `t` we compute the
F1-score on validation, and pick the `t` that maximizes it.

This trades precision against recall. If the cost of a false negative
(approving a bad loan) dominates the cost of a false positive (denying a
good applicant), the optimal threshold is lower than 0.5 -- we accept more
false alarms in exchange for catching more true defaulters.

> **Important:** threshold tuning **does not change ROC AUC**; the model
> is identical and the predicted probabilities are identical. Only the
> precision/recall/F1 numbers move.""")

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

md("""## Visual diagnostics for the best MLP

ROC curve, precision–recall curve, and the confusion matrices at the
default and tuned thresholds. Plotted on the same baseline-MLP probability
output so the only thing that changes between the two confusion matrices
is the cutoff.""")

code("""from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# ROC and PR curves on the baseline probabilities
fpr, tpr, _ = roc_curve(y_val, baseline_proba)
prec_curve, rec_curve, _ = precision_recall_curve(y_val, baseline_proba)
roc_auc_val = roc_auc_score(y_val, baseline_proba)
pr_auc_val  = average_precision_score(y_val, baseline_proba)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(fpr, tpr, color='#1F3A68', lw=2, label=f'AUC = {roc_auc_val:.4f}')
axes[0].plot([0, 1], [0, 1], '--', color='#888888', lw=1)
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC curve · best MLP (baseline probabilities)')
axes[0].legend(loc='lower right'); axes[0].grid(alpha=0.3)
for s in ('top', 'right'): axes[0].spines[s].set_visible(False)

axes[1].plot(rec_curve, prec_curve, color='#2E86C1', lw=2, label=f'PR AUC = {pr_auc_val:.4f}')
axes[1].axhline(y_val.mean(), color='#888888', linestyle='--', lw=1,
                label=f'no-skill = {y_val.mean():.4f}')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision–Recall curve · best MLP')
axes[1].legend(loc='upper right'); axes[1].grid(alpha=0.3)
for s in ('top', 'right'): axes[1].spines[s].set_visible(False)

plt.tight_layout(); plt.show()""")

code("""# Confusion matrices: default threshold (0.5) vs tuned threshold
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for ax, thr, title in [
    (axes[0], 0.5,        f'Default threshold = 0.5'),
    (axes[1], best_thr,   f'Tuned threshold = {best_thr:.4f}'),
]:
    y_pred = (baseline_proba >= thr).astype(int)
    cm = confusion_matrix(y_val, y_pred)
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['pred 0', 'pred 1'])
    ax.set_yticklabels(['true 0', 'true 1'])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    fontsize=12, fontweight='bold')

plt.tight_layout(); plt.show()""")

md("""### Quick interpretation

The numbers in the comparison table tell three different stories:

* **Baseline (threshold 0.5)** scores ROC AUC = 0.745 yet has zero recall.
  The MLP has learned a useful probability ranking, but every prediction
  sits below 0.5 -- so naively classifying at 0.5 yields a useless rule
  that flags no defaulters at all.
* **SMOTE** moves ROC AUC in the *wrong* direction (0.745 → 0.648). The
  synthetic minority points seem to push the MLP into a decision boundary
  that does worse on the real (still-imbalanced) validation set, even
  though precision and recall at threshold 0.5 are now non-zero.
* **Threshold tuning** preserves ROC AUC by construction (same model, same
  probabilities) but produces the most useful classifier of the three:
  threshold ≈ 0.16 lifts recall from 0 to ~0.40 with precision around 0.24.

The take-away on the MLP side is that **choosing the threshold matters
more than rebalancing the data** -- a finding that is consistent with
recent meta-analyses arguing against routine oversampling for properly
calibrated models.""")

# =====================================================================
# 7. Combined Results (Nathaniel Badalov + Minwoo Yoo)
# =====================================================================
md("""# Combined Results: Classic ML + Neural Networks

Loads Nathaniel Badalov's `GridSearchCV` CSVs (LR / RF / HGBC) and merges them with the
two MLP results from this notebook. This gives the full **5-model comparison**
that we present in the final slides.""")

code("""def best_row(path):
    df = pd.read_csv(path)
    df = df.sort_values('rank_test_score').head(1)
    return df['mean_test_score'].iloc[0], df['params'].iloc[0]

cv_path = result_dir + 'cv_results/GridSearchCV/'

rows = []
for fname, model_label in [
    ('lr.csv',           'Logistic Regression (Nathaniel Badalov)'),
    ('rfc.csv',          'Random Forest (Nathaniel Badalov)'),
    ('hgbc.csv',         'HistGradBoost (Nathaniel Badalov)'),
    ('shallow_mlp.csv',  'Shallow MLP (Minwoo Yoo)'),
    ('deep_mlp.csv',     'Deep MLP (Minwoo Yoo)'),
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

code("""# Best NN architecture, retrained on the full train+val data for the final
# Kaggle submission (using train+val here is fine -- no held-out evaluation).
best_mlp = MLPClassifier(
    hidden_layer_sizes=arch,
    activation='relu',
    solver='adam',
    alpha=fresh_alpha,
    learning_rate_init=fresh_lr,
    early_stopping=True,
    max_iter=100,
    random_state=random_seed,
)
best_mlp.fit(X_train_val, y_train_val)

test_proba = best_mlp.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({'SK_ID_CURR': test_ids.values, 'TARGET': test_proba})
out_path = result_dir + 'submission/submission_mlp.csv'
submission.to_csv(out_path, index=False)
print('written:', out_path)
submission.head()""")

# =====================================================================
# 9. Interpretation & Conclusion
# =====================================================================
md("""# Interpretation

### 1. Within the neural-network family

Both MLPs land essentially on top of each other:

| Architecture | Best val ROC AUC | Best params |
|---|---|---|
| Shallow MLP — hidden `(64,)`        | **0.7405** | alpha = 0.001, learning_rate_init = 0.005 |
| Deep MLP — hidden `(128, 64, 32)`   | **0.7416** | alpha = 0.001, learning_rate_init = 0.005 |

The deep MLP edges the shallow one by ~0.001 AUC -- well within the
fluctuation we would expect from a single train/val split. **Adding
depth does not help on this dataset.** That is consistent with the
"tabular bottleneck" intuition: once a flat MLP can express linear
combinations of the 244 features, extra layers mostly add capacity that
the available data cannot fill.

### 2. Compared to classic shallow learners

| Rank | Model | Best val ROC AUC | Owner |
|---|---|---|---|
| 1 | HistGradientBoosting | 0.7595 | Nathaniel Badalov |
| 2 | Logistic Regression  | 0.7487 | Nathaniel Badalov |
| 3 | Random Forest        | 0.7470 | Nathaniel Badalov |
| 4 | Deep MLP             | 0.7416 | Minwoo Yoo |
| 5 | Shallow MLP          | 0.7405 | Minwoo Yoo |

The neural networks rank **last** in the leaderboard -- about 2 AUC
points behind HGBC and on par with Logistic Regression. This matches the
broader literature on tabular data, where gradient-boosted trees
typically beat MLPs unless the dataset is very large and feature
preprocessing has been done with deep learning in mind. Two specific
factors plausibly drive the gap on this dataset:

* The 16 categorical columns become a sparse one-hot block of 150+
  features after encoding. Trees split on each binary indicator
  cheaply; the MLP has to fit weights on the whole block.
* The strongest predictors (`EXT_SOURCE_*`) are pre-engineered linear
  scores. A linear model can use them directly; a tree can split on them;
  but a deep MLP gains nothing from re-discovering the linear relationship.

### 3. Effect of imbalance handling on the MLP

| Method | Threshold | ROC AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Baseline (no handling) | 0.500   | 0.745 | 0.00 | 0.00 | 0.00 |
| SMOTE oversampling     | 0.500   | 0.648 | 0.16 | 0.21 | 0.18 |
| Threshold tuning       | 0.164   | 0.745 | 0.24 | 0.40 | 0.30 |

* **SMOTE actually hurts the MLP here.** ROC AUC drops by ~0.10 because
  the synthetic minority points distort the decision surface relative to
  the real (still-imbalanced) validation distribution.
* **Threshold tuning is the clear winner.** Same model, same probabilities,
  ROC AUC unchanged -- but the F1-optimal cutoff (~0.16) raises recall
  from 0 to 0.40. For a lender, that is the gap between "useless
  classifier" and "catches 4 in 10 defaulters".
* **The baseline rule-of-thumb 0.5 cutoff is wrong for 8% positives.**
  No prediction crossed it, so precision, recall, and F1 collapsed to
  zero even though the underlying ranker is competitive. This is the
  mechanism behind the "high AUC but useless predictions" failure mode
  that motivates the imbalance comparison.""")

md("""# Conclusion

**Best model overall.** Histogram Gradient Boosting takes the top spot at
validation ROC AUC = 0.7595, indicating that nonlinear tree-based boosting
is well suited for this dataset and can capture interactions across
heterogeneous features. Logistic Regression and Random Forest perform
similarly, both slightly below HGBC, with Logistic Regression likely
benefiting from strong linear predictors such as `EXT_SOURCE_*`. Among the
neural networks, the Deep MLP barely edges the Shallow MLP (0.7416 vs
0.7405), and both lag the classic shallow learners by ~1–2 AUC points.

**Effect of imbalance handling on the MLP.** Threshold tuning is the
right tool: it preserves AUC and lifts recall from 0% to ~40% by simply
moving the decision cutoff from 0.5 to 0.16. Resampling the training set
with SMOTE moved AUC in the wrong direction. The class_weight comparison
lives in the classic-ML notebook because sklearn's `MLPClassifier` does
not implement that parameter.

**Answer to the research question.** Yes -- shallow learners and neural
networks respond differently to imbalance handling. LR and Random Forest
benefit from `class_weight='balanced'` directly; the MLP cannot use that
hook, and the closest substitute (SMOTE) is *harmful* here. The treatment
that *does* work for the MLP -- threshold tuning -- has nothing to do
with the training data and could in principle have been applied to LR or
RF too.

**Limitations.**
* Only the main `application_*.csv` tables are used; the auxiliary
  tables (`bureau`, `previous_application`, `installments_payments`,
  `credit_card_balance`) typically close 1–2 AUC points of the gap to
  the public-leaderboard top scores.
* `PredefinedSplit` is fast but optimistic compared to k-fold CV; a full
  5-fold CV would tighten our confidence intervals.
* The MLP hyperparameter grid is small (4 combinations per architecture).
  A wider sweep over hidden sizes, dropout, and batch normalization
  would likely move the MLP up by another ~0.01 AUC, but is unlikely to
  overtake HGBC.

**Future work.**
* Integrate the four auxiliary Kaggle tables and re-run the same pipeline
  with the enriched feature set.
* Ensemble HGBC with the best MLP -- the two model families are uncorrelated
  enough that even a simple probability average tends to help.
* Calibrate probabilities (Platt or isotonic) before submitting; the
  current MLP outputs are systematically below 0.5 and would benefit
  from calibration even outside the threshold-tuning context.""")

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

out_file = 'final_project.ipynb'
with open(out_file, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'wrote {out_file} with {len(cells)} cells')
