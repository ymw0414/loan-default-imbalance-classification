# Recording Script — Final Presentation

**Total target: 8–10 minutes**
**Slides: 10**
**Speakers: Minwoo (M), Nathan (N)**

> Numbers in `{{...}}` are placeholders to be replaced after the notebook
> finishes running.

---

## Slide 1 — Title (Minwoo, ~30 s)

> Hi, I'm Minwoo Yoo, and this is my teammate Nathan. For our final project,
> we worked on the Home Credit Default Risk dataset from Kaggle. Our goal was
> to predict whether a loan applicant would default, using both classic machine
> learning models and neural networks, and we paid particular attention to how
> each model handles class imbalance.

---

## Slide 2 — Problem & Motivation (Minwoo, ~50 s)

> Lenders need to estimate the probability that an applicant will default on a
> loan, so they can balance approval volume against credit losses. The Home
> Credit dataset frames this as a binary classification problem: predict
> TARGET = 1 if the applicant defaults, 0 otherwise.

> The catch is that defaults are rare — only about eight percent of the
> applicants in our training set actually default. That makes the data heavily
> imbalanced, and a model that always predicted "no default" would still get
> 92 % accuracy while being completely useless. So our central question is not
> just "which model is most accurate," but "which model handles imbalance the
> best."

---

## Slide 3 — Dataset & EDA (Nathan, ~50 s)

> The training file has 307,511 applications described by 122 features —
> mostly numeric, with 16 categorical columns. The test set is another 48,744
> applicants whose labels are held out by Kaggle.

> Two things stood out during EDA. First, the strongest predictors are the
> three EXT_SOURCE columns — external credit scores — with correlations
> around 0.16 to 0.18 with the target. Second, several columns have very high
> missing rates: building-info features are 50 to 70 percent missing, and the
> DAYS_EMPLOYED column has a sentinel value of 365,243 that flags about 18 %
> of rows where the applicant is unemployed. We handled these with mean
> imputation for numeric columns and let one-hot encoding absorb categorical
> NaNs.

---

## Slide 4 — Research Question (Minwoo, ~40 s)

> The research question we ended up exploring is:

> *Do classic shallow models and neural networks respond differently to class
> imbalance handling techniques?*

> To answer it we set up a two-axis study. One axis is the model family — we
> compare Logistic Regression, Random Forest, Histogram Gradient Boosting,
> a shallow MLP, and a deep MLP. The other axis is the imbalance handling
> technique — baseline, class_weight = balanced, SMOTE oversampling, and
> threshold tuning.

---

## Slide 5 — Methodology (Nathan, ~60 s)

> For a fair comparison every model goes through the same preprocessing
> pipeline. We do an 80/20 stratified train–val split with random state 42 so
> the validation set keeps the same 8 % positive rate. We drop the
> SK_ID_CURR identifier, mean-impute numeric NaNs, one-hot-encode the 16
> categorical columns — that brings us up to 244 features — and finally
> StandardScale every feature, which is essential for Logistic Regression and
> for the MLPs.

> All hyperparameter tuning is done with GridSearchCV using sklearn's
> PredefinedSplit, so each model gets exactly one train→val fold instead of
> random k-fold. Our scoring metric is ROC AUC, the standard for credit-risk
> ranking problems.

---

## Slide 6 — Five Model Families (Minwoo, ~50 s)

> Here are the five models. Nathan handled the three classic ones —
> Logistic Regression and Random Forest, both with class_weight = balanced,
> and Histogram Gradient Boosting which doesn't accept that parameter. I
> handled the two neural-network variants: a shallow MLP with one 64-unit
> hidden layer, and a deep MLP with three hidden layers of 128, 64, and 32
> units. Both use ReLU activations, the Adam optimizer, and early stopping.

> The reason for this mix is to span the course curriculum — linear, tree
> bagging, tree boosting, and shallow versus deep neural networks — while
> still giving us a clean comparison axis for imbalance handling.

---

## Slide 7 — Results (Nathan, ~60 s)

> This is the combined leaderboard across all five models. The best result
> was Histogram Gradient Boosting at validation ROC AUC of 0.7595, followed
> by Logistic Regression at 0.7487, Random Forest at 0.7470, Deep MLP at
> 0.7416, and Shallow MLP at 0.7405. So gradient boosting wins by about one
> AUC point — that's consistent with the literature on tabular credit-risk
> problems, where boosted trees usually beat neural networks.

> One thing worth pointing out: Random Forest's training AUC was 0.96 while
> its validation AUC was 0.747 — a clear sign of overfitting, driven by the
> tuned leaf size of 1. And the neural networks come in essentially tied
> with the linear and bagged-tree baselines, which suggests that on this
> dataset the bottleneck is the feature set, not the model class.

---

## Slide 8 — Imbalance Handling Deep-Dive (Minwoo, ~60 s)

> For the imbalance deep-dive I take the best MLP and compare three handling
> techniques. Note that sklearn's MLPClassifier doesn't accept class_weight,
> so we cannot directly compare class-weighted training the way Nathan did
> with Logistic Regression and Random Forest.

> Baseline classifies at the default 0.5 threshold. The result is striking:
> ROC AUC is 0.745, but precision and recall are both zero — at the 0.5
> cutoff, the MLP doesn't push any prediction high enough to flag a single
> default. So the model has actually learned a useful ranking, but the
> default decision threshold is wrong for an 8 % positive rate.

> SMOTE oversampling counterintuitively makes things worse — ROC AUC drops
> from 0.745 to 0.648. The synthetic samples seem to confuse the MLP rather
> than help it.

> Threshold tuning is the clear winner. Same model as baseline, but we pick
> the decision cutoff that maximizes F1 on validation — that turns out to be
> 0.164. ROC AUC stays at 0.745 (it's the same probability scores), but
> precision goes from 0 to 0.24 and recall from 0 to 0.40. So for an
> imbalanced problem like this, *picking the right threshold* matters more
> than *fixing the class balance during training*.

---

## Slide 9 — Conclusions (Minwoo, ~50 s)

> Three takeaways. First, Histogram Gradient Boosting is the best single
> model at 0.7595 AUC, beating both linear and neural networks by about one
> AUC point — consistent with the literature on tabular data. Second, our
> neural networks land essentially tied with Logistic Regression and Random
> Forest, which suggests that on this dataset the bottleneck is the feature
> set rather than the model class. Going from a shallow MLP to a deep MLP
> moves AUC by 0.001 — basically noise. Third, the most actionable lesson
> from the imbalance comparison is that *threshold matters more than
> resampling*: keeping the baseline model and just choosing a sensible
> threshold turned a 0-recall classifier into one that catches 40 % of
> defaults.

> Limitations: we only used the main application_train and application_test
> tables. The auxiliary tables — bureau, previous_application, installments —
> are known to add several AUC points. And our PredefinedSplit gives a fast
> but optimistic estimate; full k-fold CV would tighten the confidence
> intervals.

---

## Slide 10 — Q&A (Both, ~10 s)

> That's our project. The notebook and slides are on the GitHub repo shown
> here. Thanks for watching, and we're happy to take any questions.

---

# Recording tips

- **Resolution:** record at 1080p, 16:9.
- **Microphone:** use a headset mic; test levels first.
- **Software (free options):**
  - **OBS Studio** — most professional; record screen + mic.
  - **Zoom** — start a meeting alone, "Record to this computer," screen-share
    your slides.
  - **PowerPoint built-in** — *Slide Show → Record* records voice over slides
    directly and exports to MP4.
- **Pacing:** 10 slides / 9 min ≈ 54 s per slide. Don't sprint.
- **NG handling:** if you stumble, pause, take a breath, restart from the
  *previous slide title* (clean cut point in editing). Most tools (PowerPoint,
  Zoom) let you re-record a single slide without redoing the whole deck.
- **Upload target:** YouTube as **Unlisted** (not Public, not Private — the
  professor has to be able to open it without an account). Paste the link
  into Slide 10 before submission.

# Pre-recording checklist

- [ ] Notebook ran end-to-end without errors
- [ ] All `{{...}}` placeholders in this script replaced with real numbers
- [ ] Slide deck PDF exported (some platforms don't render fonts the same)
- [ ] Recording link inserted into Slide 10
- [ ] Final files staged: `final_project_minwoo.ipynb`, `final_presentation.pptx`,
      recording link
- [ ] All three uploaded to Blackboard before 23:59 ET on 04/27
