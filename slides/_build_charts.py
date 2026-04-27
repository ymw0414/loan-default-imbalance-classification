"""
Builds the chart PNGs that get embedded into the slide deck.

Reads from   ../result/home_credit/...
Writes to    ../result/home_credit/figure/

Charts produced:
  - model_auc_bar.png      : 5-model bar chart of validation ROC AUC
  - imbalance_compare.png  : grouped bar of (precision, recall, F1) per technique
"""
from __future__ import annotations

import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE          = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(HERE, '..'))
CV_DIR        = os.path.join(PROJECT_ROOT, 'result', 'home_credit', 'cv_results', 'GridSearchCV')
COMBINED_CSV  = os.path.join(PROJECT_ROOT, 'result', 'home_credit', 'combined_model_comparison.csv')
IMBAL_CSV     = os.path.join(CV_DIR, 'imbalance_compare.csv')
FIG_DIR       = os.path.join(PROJECT_ROOT, 'result', 'home_credit', 'figure')

os.makedirs(FIG_DIR, exist_ok=True)

# Visual identity (keep close to the slide deck colors)
TITLE_COLOR = '#1F3A68'
ACCENT      = '#2E86C1'
GREY        = '#555555'
PALETTE     = ['#1F3A68', '#2E86C1', '#5DADE2', '#7FB3D5', '#A9CCE3']


# --------------------------------------------------------------------- chart 1
def chart_model_bar() -> str | None:
    """Horizontal bar of the five models' best validation ROC AUC."""

    # Prefer the combined CSV; fall back to per-model CSVs.
    if os.path.exists(COMBINED_CSV):
        df = pd.read_csv(COMBINED_CSV)
    else:
        rows = []
        for fname, label in [
            ('lr.csv',          'Logistic Regression'),
            ('rfc.csv',         'Random Forest'),
            ('hgbc.csv',        'HistGradBoost'),
            ('shallow_mlp.csv', 'Shallow MLP'),
            ('deep_mlp.csv',    'Deep MLP'),
        ]:
            p = os.path.join(CV_DIR, fname)
            if os.path.exists(p):
                d = pd.read_csv(p).sort_values('rank_test_score').head(1)
                rows.append({'model': label, 'best_val_AUC': float(d['mean_test_score'].iloc[0])})
        df = pd.DataFrame(rows)

    df = df.dropna(subset=['best_val_AUC']).sort_values('best_val_AUC')

    if df.empty:
        print('chart_model_bar: no data yet, skipping')
        return None

    # Strip trailing "(Nathan)" / "(Minwoo)" tags for the chart axis
    df = df.copy()
    df['display'] = df['model'].astype(str).str.replace(r'\s*\(.*\)$', '', regex=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=160)
    bars = ax.barh(df['display'], df['best_val_AUC'], color=PALETTE[: len(df)])
    ax.set_xlabel('Validation ROC AUC', color=GREY)
    ax.set_title('Model comparison · best validation ROC AUC', color=TITLE_COLOR, fontsize=14, weight='bold')
    ax.set_xlim(0.70, max(df['best_val_AUC'].max() + 0.02, 0.78))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for bar, score in zip(bars, df['best_val_AUC']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{score:.4f}', va='center', fontsize=10, color=TITLE_COLOR)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'model_auc_bar.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)
    return out


# --------------------------------------------------------------------- chart 2
def chart_imbalance_bar() -> str | None:
    """Grouped bar chart of precision / recall / F1 per imbalance technique."""

    if not os.path.exists(IMBAL_CSV):
        print('chart_imbalance_bar: imbalance CSV missing, skipping')
        return None

    df = pd.read_csv(IMBAL_CSV)
    if df.empty:
        return None

    methods = df['method'].tolist()
    metrics = ['precision', 'recall', 'F1']
    n_methods, n_metrics = len(methods), len(metrics)

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=160)
    x = range(n_methods)
    bar_w = 0.25

    for j, m in enumerate(metrics):
        ax.bar([xi + (j - 1) * bar_w for xi in x],
               df[m].values, width=bar_w, label=m,
               color=PALETTE[j])

    ax.set_xticks(list(x))
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score', color=GREY)
    ax.set_title('Imbalance handling · precision / recall / F1 on best MLP',
                 color=TITLE_COLOR, fontsize=14, weight='bold')
    ax.legend(frameon=False, loc='upper left')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    # Value labels on each bar
    for j, m in enumerate(metrics):
        for i, v in enumerate(df[m].values):
            ax.text(i + (j - 1) * bar_w, v + 0.01, f'{v:.2f}',
                    ha='center', fontsize=8, color=GREY)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'imbalance_compare.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)
    return out


def chart_class_balance() -> str:
    """Static chart for the EDA slide showing the 8% / 92% class imbalance."""

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=180)
    classes = ['No default\n(TARGET = 0)', 'Default\n(TARGET = 1)']
    counts  = [282686, 24825]
    pct     = [c / sum(counts) * 100 for c in counts]
    bars = ax.barh(classes, counts, height=0.55, color=['#1A365D', '#D97706'])
    for bar, c, p in zip(bars, counts, pct):
        ax.text(bar.get_width() + 4000, bar.get_y() + bar.get_height() / 2,
                f'{c:,}  ({p:.2f}%)', va='center', fontsize=14, weight='bold',
                color='#1A365D')
    ax.set_xlim(0, 340000)
    ax.set_xlabel('Number of applicants', fontsize=11, color='#555555')
    ax.set_title('Severe class imbalance — only 8% positive',
                 fontsize=15, weight='bold', color='#1A365D', pad=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=10)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'class_balance.png')
    plt.savefig(out, bbox_inches='tight', dpi=180)
    plt.close(fig)
    print('wrote', out)
    return out


if __name__ == '__main__':
    chart_class_balance()
    chart_model_bar()
    chart_imbalance_bar()
    print('done')
