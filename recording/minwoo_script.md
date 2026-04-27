# Minwoo's Recording Script — 5 minutes

**Total: 5:00 minutes  ·  4 slides + half of closing slide**

> Slides Minwoo presents: **1, 2, 3, 7, 8** (and joins on slide 9, 10).
> Slides Nathan presents: **4, 5, 6** (and joins on slide 9, 10).
>
> All numbers below come from the actual notebook run on 2026-04-27.
> If you re-run the notebook, refresh this script before recording.

---

## SLIDE 1 — Title (≈ 0:30)

> Hi, I'm Minwoo Yoo, and this is my teammate Nathan. For our final project we
> tackled the **Home Credit Default Risk** dataset from Kaggle. The goal is to
> predict whether a loan applicant will default — a binary classification
> problem — and we paid particular attention to how different model families
> respond to **class imbalance**.

*(Hand off briefly: "Let me start with why this problem is hard.")*

---

## SLIDE 2 — Problem & Motivation (≈ 1:00)

> Lenders need to estimate the probability that an applicant will default, so
> they can balance approval volume against credit losses. The Home Credit
> dataset frames this as binary classification — predict `TARGET = 1` if the
> applicant defaults, `0` otherwise.

> The catch is that defaults are rare — only about **8 percent** of training
> applicants actually default. That makes the data heavily imbalanced.
> A naive model that always predicted "no default" would still get **92 percent
> accuracy** while being completely useless.

> Even more striking, as we'll see later, our trained MLP reaches a respectable
> ROC AUC of 0.745 — yet at the default 0.5 cutoff it classifies **zero
> applicants** as defaulters. So our central question wasn't just "which model
> is most accurate," but:

> *"Do classic shallow learners and neural networks respond differently to
> class imbalance handling?"*

---

## SLIDE 3 — Dataset · Home Credit (≈ 1:00)

> The training file has **307,511 applicants** described by 122 features.
> The test set is another 48,744 applicants whose labels Kaggle keeps hidden.
> Default rate in the training data is exactly the 8.07% I just mentioned.

> Three things stood out during EDA. First, the strongest predictors are the
> three **`EXT_SOURCE`** columns — external credit scores — with correlation
> magnitudes around 0.16 to 0.18. Second, **`DAYS_EMPLOYED`** has a sentinel
> value of 365,243 that flags about 18 percent of rows where the applicant is
> unemployed. And third, building-info columns are 50 to 70 percent missing —
> we mean-impute the numeric ones and let one-hot encoding absorb categorical
> NaNs.

> *(Pause briefly, then hand off:)* That covers the data. Nathan will walk you
> through the shared pipeline and the classic-ML side.

**[HAND-OFF TO NATHAN]**

*(Nathan presents slides 4, 5, 6 — methodology, five models lineup, combined
ROC AUC results — for about 4 minutes.)*

---

## SLIDE 7 — Neural Network Track (≈ 1:30)

*(Take back from Nathan with a short bridge:)*

> Thanks Nathan. So the classic shallow learners give us a clear baseline.
> Now let me walk through the neural-network side.

> I built two MLPs, both with `sklearn`'s `MLPClassifier`. The **shallow MLP**
> has one hidden layer of 64 neurons — about 15 thousand parameters. The
> **deep MLP** has three hidden layers — 128, 64, 32 — with about 42 thousand
> parameters, roughly three times as many. Both use ReLU activations, the Adam
> optimizer, and `early_stopping=True` so training halts automatically when
> validation loss stops improving.

> Hyperparameter tuning was done with the same `GridSearchCV` and `PredefinedSplit`
> pipeline Nathan described, over `alpha` — the L2 regularization strength —
> and the initial Adam learning rate. The best setting for both architectures
> turned out to be `alpha = 1e-3` and `learning_rate_init = 0.005`.

> Now the headline: the **shallow MLP** scored **0.7405** validation ROC AUC.
> The **deep MLP** scored **0.7416**. That's a difference of 0.001 AUC — well
> within noise. So tripling the parameter count bought us essentially nothing.
> On this dataset, going deeper does **not** help, and both NNs land slightly
> below the classic shallow learners.

> The interpretation is straightforward: tabular data with mostly hand-crafted
> features doesn't benefit from the kind of representation learning neural
> networks excel at. The bottleneck is the feature set, not the model class.

---

## SLIDE 8 — Imbalance Handling Deep Dive (≈ 1:30)

> But the most interesting part of the project was the **imbalance handling
> comparison**. `MLPClassifier` doesn't accept `class_weight`, so I compared
> three other strategies on the best MLP.

> **Strategy 1, baseline.** Train as is, classify at threshold 0.5. The result
> is striking — ROC AUC is 0.745, so the model has clearly learned a useful
> probability ranking. But precision and recall are both **zero**. Why? Because
> the model's probability outputs never cross 0.5 — at that cutoff it predicts
> "no default" for every single applicant in the validation set. Useful ranker,
> useless classifier.

> **Strategy 2, SMOTE oversampling.** Synthesize minority-class samples by
> interpolating between existing default points until the training set is
> 50/50, then re-train. Counterintuitively, this **hurt** the model — ROC AUC
> dropped from 0.745 to 0.648, a loss of about 10 AUC points. The synthetic
> minority samples seem to push the MLP into a decision boundary that does
> worse on the real, still-imbalanced validation set.

> **Strategy 3, threshold tuning.** Keep the baseline model untouched, but
> instead of using 0.5 as the cutoff, sweep all thresholds and pick the one
> that maximizes F1 on validation. That cutoff turns out to be **0.164**.
> Threshold tuning preserves ROC AUC by construction — we haven't touched the
> probabilities — but it lifts recall from **zero to 40 percent**, with
> precision around 0.24 and F1 of 0.30.

> So the headline finding for the MLP is: **choosing the threshold matters
> more than rebalancing the data.** SMOTE made things worse; threshold tuning
> turned a useless classifier into one that catches four out of every ten
> defaulters. That's the takeaway I'd like you to remember.

> *(Brief pause, then:)* Nathan, take it home with the conclusions.

**[HAND-OFF TO NATHAN — slide 9 conclusions]**

---

## SLIDE 10 — Closing (≈ 0:15, joint)

*(Both presenters on screen for the final slide.)*

**Minwoo:** Thanks for watching.

**Nathan:** The recording link, our GitHub, and team info are on this slide.

*(Brief beat, end recording.)*

---

# Speaker time accounting

| Slide | Speaker | Duration |
|---|---|---|
| 1     | Minwoo  | 0:30 |
| 2     | Minwoo  | 1:00 |
| 3     | Minwoo  | 1:00 |
| 4     | Nathan  | 1:00 |
| 5     | Nathan  | 1:30 |
| 6     | Nathan  | 1:00 |
| 7     | Minwoo  | 1:30 |
| 8     | Minwoo  | 1:30 |
| 9     | Nathan  | 1:00 |
| 10    | Both    | 0:15 |
| **Total** |     | **10:15** |

| Speaker | Total |
|---|---|
| **Minwoo** | **5:30**  (slides 1, 2, 3, 7, 8) |
| **Nathan** | **4:30**  (slides 4, 5, 6, 9) |
| **Joint**  | **0:15**  (slide 10) |

> Total runs slightly long. To trim to 10:00, shorten Slide 8 by ~20 seconds —
> drop the "ROC AUC by construction" parenthetical and one of the strategy
> recaps. Or shorten Slide 5 by ~15 seconds.

---

# Tips for Minwoo's portion

## Pacing
- **Slide 2 is the most rhetorically important** — the 0% recall hook.
  Don't rush; let the "92% accuracy yet zero recall" land.
- **Slide 8 is the climax** — the threshold-vs-SMOTE finding is the
  single most memorable thing in the talk. Pause before the "choosing
  the threshold matters more than rebalancing the data" line.

## Voice / delivery
- Read once aloud at full speed before recording — feel where you naturally
  pause.
- If you stumble, don't say "uh, sorry" — just stop, breathe, and start the
  same sentence over. You can edit the false start out (or, in PowerPoint /
  Zoom recording, just re-record the slide).

## Slide handover phrasing
- End of Slide 3: *"Nathan will walk you through the shared pipeline and the
  classic-ML side."*
- Start of Slide 7: *"Thanks Nathan. So the classic shallow learners give us
  a clear baseline. Now let me walk through the neural-network side."*
- End of Slide 8: *"Nathan, take it home with the conclusions."*

These verbal handoffs make the transition feel intentional rather than abrupt
— important for a two-presenter recording.

## What to do if you misspeak a number
- Threshold = 0.164 (often misread as 0.16 or 0.1641)
- Recall lifted to **40 percent** (or "0.40", not "40 something")
- Best AUC = 0.7595 (Nathan will say this; you'll mention 0.745 for the MLP)

If you say a wrong number, just say "let me restate that" and continue —
do not try to make a correction sound graceful in real time.
