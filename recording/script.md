# Recording Script — Final Presentation

**Total target: 8–10 minutes  ·  10 slides  ·  Two presenters**

| Speaker | Slides | Duration |
| :--- | :--- | :--- |
| **Nathaniel Badalov**  (front half) | 1, 2, 3, 4, 5 | ~5 min |
| **Minwoo Yoo**  (back half)        | 6, 7, 8, 9    | ~5 min |
| Joint                               | 10            | ~15 s   |

> All numbers below come from the actual notebook run on 2026-04-27.
> If you re-run the notebook, refresh this script with the latest values
> before recording.

---

## SLIDE 1 — Title  *(Nathaniel · 0:30)*

> Hi, I'm Nathaniel Badalov, and this is my teammate Minwoo Yoo. For our
> final project we tackled the **Home Credit Default Risk** dataset from
> Kaggle. The goal is to predict whether a loan applicant will default —
> a binary classification problem — with particular attention to how
> different model families respond to **class imbalance**.

---

## SLIDE 2 — Problem & Motivation  *(Nathaniel · 1:00)*

> Lenders need to estimate the probability that an applicant will default,
> so they can balance approval volume against credit losses. The Home
> Credit dataset frames this as binary classification — predict
> `TARGET = 1` if the applicant defaults, `0` otherwise.

> The catch is that defaults are rare — only about **8 percent** of training
> applicants actually default. That makes the data heavily imbalanced.
> A naive "always predict no default" model would still score
> 92 percent accuracy while being completely useless. As we'll see later,
> even a trained MLP with a respectable ROC AUC of 0.745 ends up
> classifying **zero** applicants as defaulters at the default 0.5 cutoff.

> So our central question is:
> *Do classic shallow learners and neural networks respond differently to
> class imbalance handling techniques?*

---

## SLIDE 3 — Dataset · Home Credit  *(Nathaniel · 1:00)*

> The training file has **307,511 applicants** described by 122 features.
> The test set is another 48,744 applicants whose labels Kaggle keeps
> hidden. Default rate in the training data is exactly the 8.07% I just
> mentioned.

> Three things stood out during EDA. First, the strongest predictors are
> the three **EXT_SOURCE** columns — external credit scores — with
> correlation magnitudes around 0.16 to 0.18. Second, **DAYS_EMPLOYED**
> has a sentinel value of 365,243 that flags about 18 percent of rows
> where the applicant is unemployed. Third, building-info columns are
> 50 to 70 percent missing — we mean-impute the numeric ones and let
> one-hot encoding absorb categorical NaNs.

---

## SLIDE 4 — Methodology · Shared Pipeline  *(Nathaniel · 1:00)*

> Every model goes through the same preprocessing pipeline so the
> comparison is fair. We do an 80/20 stratified train–val split with
> random state 42 — that keeps the 8 percent positive rate identical in
> both halves. We drop the SK_ID_CURR identifier, mean-impute numeric
> NaNs, one-hot-encode the 16 categorical columns — bringing us to
> **244 features** — and finally StandardScale every feature, which is
> essential for Logistic Regression and the MLPs.

> Hyperparameter tuning is done with `GridSearchCV` using sklearn's
> `PredefinedSplit`, so each model gets exactly one train→val fold
> instead of random k-fold. Our scoring metric is ROC AUC, the standard
> for credit-risk ranking problems.

---

## SLIDE 5 — Five Models Compared  *(Nathaniel · 1:30)*

> Here are the five models. I built the three classic shallow learners on
> the left of the table — Logistic Regression and Random Forest, both
> with `class_weight='balanced'`, and HistGradientBoosting which
> doesn't accept that parameter. Minwoo built the two neural networks
> on the right — a shallow MLP with one 64-unit hidden layer, and a deep
> MLP with three hidden layers of 128, 64, and 32 units. Both NNs use
> ReLU activations, the Adam optimizer, and early stopping.

> The reason for this lineup is to span the course curriculum — linear,
> tree bagging, tree boosting, and shallow versus deep neural networks —
> while keeping a clean comparison axis: same preprocessing, same
> train/val split, same scoring metric.

> Now Minwoo will walk you through how the five models compared in
> validation AUC.

**[HAND-OFF TO MINWOO]**

---

## SLIDE 6 — Combined Validation ROC AUC  *(Minwoo · 1:00)*

> Thanks Nathaniel. Here's the combined leaderboard across all five
> models. **HistGradientBoosting wins** with a validation ROC AUC of
> **0.7595**, followed by Logistic Regression at 0.7487, Random Forest at
> 0.7470, Deep MLP at 0.7416, and Shallow MLP at 0.7405. So gradient
> boosting wins by about one AUC point — that's consistent with the
> literature on tabular credit-risk problems, where boosted trees
> typically beat neural networks.

> Two quick observations. First, Random Forest's training AUC was 0.96
> while its validation AUC was 0.747 — a clear sign of overfitting,
> driven by the tuned leaf size of 1. Second, the neural networks
> come in essentially tied with the linear and bagged-tree baselines,
> which suggests that on this dataset the bottleneck is the feature
> set, not the model class.

---

## SLIDE 7 — Neural Network Track  *(Minwoo · 1:30)*

> Let me zoom in on the neural-network side, which is what I built. The
> shallow MLP has one hidden layer of 64 neurons — about 15 thousand
> parameters. The deep MLP has three hidden layers — 128, 64, 32 — with
> about 42 thousand parameters, roughly three times as many.

> Hyperparameter tuning was over `alpha`, the L2 regularization strength,
> and the initial Adam learning rate. The best setting for both
> architectures turned out to be `alpha = 1e-3` and
> `learning_rate_init = 0.005`.

> The headline: shallow scored **0.7405**, deep scored **0.7416** — a
> difference of 0.001 AUC, well within noise. Tripling the parameter
> count bought us nothing. On this dataset, going deeper does **not**
> help. The interpretation is straightforward: tabular data with mostly
> hand-crafted features doesn't benefit from the kind of representation
> learning neural networks excel at — the bottleneck really is the
> feature set, not the model class.

---

## SLIDE 8 — Imbalance Handling Deep Dive  *(Minwoo · 1:30)*

> But the most interesting part of the project was the **imbalance
> handling comparison**. `MLPClassifier` doesn't accept `class_weight`,
> so I compared three other strategies on the best MLP.

> **Strategy 1, baseline.** Train as is, classify at 0.5. The result is
> striking — ROC AUC is 0.745, so the model has clearly learned a useful
> probability ranking. But precision and recall are both **zero**.
> Why? Because the model's probability outputs never cross 0.5 — at that
> cutoff it predicts "no default" for every applicant. Useful ranker,
> useless classifier.

> **Strategy 2, SMOTE oversampling.** Synthesize minority-class samples
> by interpolating between existing default points until the training
> set is 50/50, then re-train. Counterintuitively, this **hurt** the
> model — ROC AUC dropped from 0.745 to 0.648, a loss of about 10 AUC
> points.

> **Strategy 3, threshold tuning.** Keep the baseline model untouched,
> but instead of using 0.5 sweep all thresholds and pick the one that
> maximizes F1 on validation. That cutoff is **0.164**. Threshold tuning
> preserves ROC AUC by construction, but it lifts recall from
> **zero to 40 percent**, with precision around 0.24 and F1 of 0.30.

> So the headline finding: **choosing the threshold matters more than
> rebalancing the data.** SMOTE made things worse; threshold tuning
> turned a useless classifier into one that catches four out of every
> ten defaulters. That's the takeaway I'd like you to remember.

---

## SLIDE 9 — Conclusions  *(Minwoo · 0:45)*

> Three takeaways. First, HistGradientBoosting at 0.7595 AUC is the best
> single model — tree boosting wins by about one AUC point. Second,
> neural networks land essentially tied with Logistic Regression and
> Random Forest; depth alone doesn't unlock new performance. Third, on
> the MLP, threshold tuning is the right tool for imbalance — recall
> jumped from 0% to 40%, while SMOTE actually moved AUC in the wrong
> direction.

> Limitations: we only used the main `application_*.csv` tables.
> Auxiliary tables — bureau, previous_application, installments — would
> typically add several AUC points. And our PredefinedSplit is fast but
> optimistic; full k-fold CV would tighten the confidence intervals.

---

## SLIDE 10 — Closing  *(Both · 0:15)*

**Minwoo:** Thanks for watching.

**Nathaniel:** The recording link, our GitHub, and team info are on
this slide.

*(End recording.)*

---

# Time accounting

| Slide | Speaker | Duration |
|---|---|---|
| 1     | Nathaniel | 0:30 |
| 2     | Nathaniel | 1:00 |
| 3     | Nathaniel | 1:00 |
| 4     | Nathaniel | 1:00 |
| 5     | Nathaniel | 1:30 |
| 6     | Minwoo    | 1:00 |
| 7     | Minwoo    | 1:30 |
| 8     | Minwoo    | 1:30 |
| 9     | Minwoo    | 0:45 |
| 10    | Both      | 0:15 |
| **Total** |       | **10:00** |

| Speaker | Total |
|---|---|
| **Nathaniel Badalov** | **5:00** (slides 1–5) |
| **Minwoo Yoo**        | **4:45** (slides 6–9) |
| **Joint**             | **0:15** (slide 10)   |

---

# Tips

## Pacing
* **Slide 2** — let the "92% accuracy yet zero recall" line land. Don't rush.
* **Slide 8** — the threshold-vs-SMOTE finding is the single most memorable
  thing in the talk. Pause before the "choosing the threshold matters more
  than rebalancing the data" line.

## Hand-off phrasing (built into the script above)
* End of Slide 5 (Nathaniel): *"Now Minwoo will walk you through how the
  five models compared in validation AUC."*
* Start of Slide 6 (Minwoo): *"Thanks Nathaniel. Here's the combined
  leaderboard..."*

These verbal cues make the transition feel intentional rather than abrupt —
important for a two-presenter recording.

## Recording mechanics
* **Record at 1080p, 16:9.** Use a headset mic; test levels first.
* **Free options:** PowerPoint *Slide Show → Record*, Zoom (start a meeting
  alone, "Record to this computer"), OBS Studio.
* **NG handling:** if you stumble, pause, breathe, restart from the
  *previous slide title* (clean cut point in editing). Most tools let you
  re-record a single slide without redoing the whole deck.
* **Upload as YouTube Unlisted** (not Public, not Private — the professor
  has to be able to open it without an account). Paste the link into
  Slide 10 before the final compile.

## Pre-recording checklist
- [ ] Notebook executes cleanly end-to-end
- [ ] All numbers in this script match the latest run
- [ ] Slide deck PDF compiles with no missing figures
- [ ] Recording link inserted into Slide 10 of `final_presentation.tex`
- [ ] PDF re-compiled with the recording link
- [ ] Final files staged: `final_project.ipynb`, `final_presentation.pdf`,
      recording link
- [ ] All uploaded to Blackboard before 23:59 ET on 2026-04-27
