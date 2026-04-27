# Recording Script — Final Presentation

**Total target: ~10 minutes  ·  10 slides  ·  Two presenters**

| Speaker | Slides | Duration |
| :--- | :--- | :--- |
| **Nathaniel Badalov**  (intro · data · classic ML) | 1, 2, 3, 4, 5 | ~5 min |
| **Minwoo Yoo**  (NN · combined · imbalance · conclusion) | 6, 7, 8, 9 | ~5 min |
| Joint                                              | 10           | ~15 s   |

> All numbers below come from the actual notebook run on 2026-04-27.

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
> so they can balance approval volume against credit losses. False
> negatives — approving a borrower who later defaults — cost the full
> unpaid principal, while denying a creditworthy borrower only forfeits
> the interest margin. So the cost is asymmetric, and we need a model
> that ranks applicants well.

> The catch is that defaults are rare — only about **8 percent** of
> training applicants actually default. That makes the data heavily
> imbalanced. A naive "always predict no default" model would still
> score 92 percent accuracy while being completely useless. Because of
> this, we evaluate using **ROC AUC** rather than raw accuracy.

> So our central question:
> *Do classic shallow learners and neural networks respond differently
> to class imbalance handling techniques?*

---

## SLIDE 3 — Dataset · Home Credit  *(Nathaniel · 1:00)*

> The training file has **307,511 applicants** described by 122 features.
> The test set is another 48,744 applicants whose labels Kaggle keeps
> hidden. Default rate in training is exactly the 8.07% I just mentioned.

> Three things stood out during EDA. First, the strongest predictors are
> the three **EXT_SOURCE** columns — external credit scores — with
> correlation magnitudes around 0.16 to 0.18. Second, **DAYS_EMPLOYED**
> has a sentinel value of 365,243 that flags about 18 percent of rows
> where the applicant is unemployed. Third, building-info columns are
> 50 to 70 percent missing — we mean-impute the numeric ones and let
> one-hot encoding absorb categorical NaNs.

---

## SLIDE 4 — Shared Pipeline  *(Nathaniel · 1:00)*

> Every model goes through the same preprocessing pipeline so the
> comparison is fair. We do an 80/20 stratified train–val split with
> random state 42 — keeping the 8% positive rate identical in both
> halves. We drop the SK_ID_CURR identifier, mean-impute numeric NaNs,
> one-hot-encode the 16 categorical columns — bringing us up to **244
> features** — and StandardScale every feature, which is essential for
> Logistic Regression and the MLPs.

> Hyperparameter tuning is done with `GridSearchCV` using sklearn's
> `PredefinedSplit`, so each model gets one train→val fold instead of
> random k-fold. Our scoring metric is ROC AUC.

---

## SLIDE 5 — Classic ML Models  *(Nathaniel · 1:30)*

> I built three classic shallow learners covering the main tabular ML
> families.

> **Logistic Regression** with `class_weight='balanced'`, tuned over
> regularization strength `C` and tolerance `tol`. Best val AUC = 0.7487.

> **Random Forest**, also with `class_weight='balanced'`, tuned over the
> number of trees, minimum samples per split, and minimum leaf size.
> Best val AUC = 0.7470. One thing worth noting — when min_samples_leaf
> is 1, train AUC reaches 0.96 while val stays at 0.747; that's a clear
> overfitting signal, and the regularized config wins.

> **HistGradientBoosting** — sklearn's fast gradient-boosted-tree
> implementation, tuned over learning rate, number of iterations, and
> minimum leaf size. Best val AUC = **0.7595** — the strongest of all
> three classic models.

> So gradient boosting wins among the classic learners, with linear
> models surprisingly close behind. That's consistent with the
> literature on tabular credit data, where boosted trees tend to
> dominate.

> *(Hand-off cue:)* Now Minwoo Yoo will walk you through the neural network
> side.

**[HAND-OFF TO MINWOO]**

---

## SLIDE 6 — Neural Network Models  *(Minwoo Yoo · 1:30)*

> Thanks Nathaniel. Let me walk through the neural-network side, which
> is what I built. I trained two MLPs using sklearn's `MLPClassifier`.

> The **shallow MLP** has one hidden layer of 64 neurons — about 15
> thousand parameters. The **deep MLP** has three hidden layers — 128,
> 64, 32 — with about 42 thousand parameters, roughly three times as
> many.

> Both used ReLU activations, the Adam optimizer, and `early_stopping`
> so training halts automatically when validation loss stops improving.
> I tuned over `alpha` — the L2 regularization strength — and the
> initial Adam learning rate. The best for both turned out to be
> `alpha = 1e-3` and `learning_rate_init = 0.005`.

> The headline: shallow scored **0.7405**, deep scored **0.7416** — a
> difference of 0.001 AUC, well within noise. Tripling the parameter
> count bought us essentially nothing. On this dataset, going deeper
> does **not** help. Both NNs land slightly below Nathaniel's classic
> shallow learners — the bottleneck is the feature set, not the model
> class.

---

## SLIDE 7 — Combined Leaderboard  *(Minwoo Yoo · 1:00)*

> Here's the combined leaderboard across all five models.
> **HistGradientBoosting wins** at 0.7595 validation AUC, followed by
> Logistic Regression at 0.7487, Random Forest at 0.7470, Deep MLP at
> 0.7416, and Shallow MLP at 0.7405. Tree boosting wins by about one AUC
> point — consistent with the literature on tabular data, where boosted
> trees typically beat neural networks.

> The neural networks finish essentially tied with the linear and
> bagged-tree baselines. So the answer to "is more model capacity
> better?" on this dataset is "no, not really".

---

## SLIDE 8 — Imbalance Handling Deep Dive  *(Minwoo Yoo · 1:30)*

> But the most interesting part of the project was the imbalance
> handling comparison on the MLP. `MLPClassifier` doesn't accept
> `class_weight`, so I compared three other strategies on the best NN.

> **Strategy 1, baseline.** Train as is, classify at threshold 0.5.
> ROC AUC is 0.742 — the model has clearly learned a useful probability
> ranking. But precision and recall are both **zero**. Why? Because the
> model's probability outputs never cross 0.5. Useful ranker, useless
> classifier.

> **Strategy 2, SMOTE oversampling.** Synthesize minority-class samples
> until the training set is 50/50, then re-train. Counterintuitively,
> this **hurt** the model — ROC AUC dropped from 0.742 to 0.648.

> **Strategy 3, threshold tuning.** Keep the baseline model, but pick
> the threshold that maximizes F1 on validation. That cutoff is
> **0.139**. Threshold tuning preserves ROC AUC — the model is identical
> — but it lifts recall from **zero to 41 percent**, with precision
> around 0.22 and F1 of 0.29.

> So the headline finding for the MLP is: **choosing the threshold
> matters more than rebalancing the data.** SMOTE made things worse;
> threshold tuning turned a useless classifier into one that catches
> four out of every ten defaulters.

---

## SLIDE 9 — Conclusions  *(Minwoo Yoo · 0:45)*

> Three takeaways. First, HistGradientBoosting at 0.7595 AUC is the best
> single model — tree boosting wins by about one AUC point. Second,
> neural networks tie LR and Random Forest; depth alone doesn't unlock
> new performance on this tabular data. Third, on the MLP, threshold
> tuning is the right tool for imbalance — recall jumped from 0% to
> 41%, while SMOTE actually moved AUC in the wrong direction.

> Limitations: we only used the main `application_*.csv` tables.
> Auxiliary tables — `bureau`, `previous_application`,
> `installments_payments` — would typically add several AUC points.
> And our PredefinedSplit is fast but optimistic; full k-fold CV would
> tighten confidence intervals.

---

## SLIDE 10 — Closing  *(Both · 0:15)*

**Minwoo Yoo:** Thanks for watching.

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
| 6     | Minwoo Yoo    | 1:30 |
| 7     | Minwoo Yoo    | 1:00 |
| 8     | Minwoo Yoo    | 1:30 |
| 9     | Minwoo Yoo    | 0:45 |
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
* **Slide 5** — Nathaniel: 1:30 is enough to give each of the 3 models a
  20-second beat, plus 30 seconds for findings. Don't rush the AUC numbers.
* **Slide 8** — Minwoo Yoo: the threshold-vs-SMOTE finding is the single most
  memorable thing in the talk. Pause before "*choosing the threshold matters
  more than rebalancing the data.*"

## Hand-off phrasing (built into the script)
* End of Slide 5 (Nathaniel): *"Now Minwoo Yoo will walk you through the neural
  network side."*
* Start of Slide 6 (Minwoo Yoo): *"Thanks Nathaniel. Let me walk through the
  neural-network side, which is what I built."*

## Pre-recording checklist
- [ ] Notebook executes cleanly end-to-end
- [ ] All numbers in this script match the latest run
- [ ] Slide deck PDF compiles with no missing figures
- [ ] Recording link inserted into Slide 10 of `final_presentation.tex`
- [ ] PDF re-compiled with the recording link
- [ ] Final files staged: `final_project.ipynb`, `final_presentation.pdf`,
      recording link
- [ ] All uploaded to Blackboard before 23:59 ET on 2026-04-27
