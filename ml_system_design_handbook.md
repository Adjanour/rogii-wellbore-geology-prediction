# ML System Design — A Practitioner's Handbook
*From first principles to production-ready thinking*

---

## Preface

Most ML education teaches you the middle of the story — algorithms, loss functions, hyperparameters. This handbook teaches you the edges: the thinking that happens *before* you touch data and *after* you get a prediction. It is grounded in real problems, not toy examples.

The central argument is simple: **the algorithm is usually the least interesting decision you will make.** The decisions about operational reality, data availability at inference time, feature representation, and output form matter more — and they are made before a single line of training code is written.

---

## Part I — The Mental Model

### 1.1 What a model actually is

A model is a function approximator.

You have some input X and some output Y. The true relationship between them — the function that maps X to Y — is either unknown, too complex to write explicitly, or too slow to compute directly. A model learns an approximation of that function by observing many (X, Y) pairs and adjusting its internal parameters until its approximation is close enough to be useful.

The learning algorithm — gradient boosting, backpropagation, random forests — is the mechanism that does the adjusting. It is not the model. The model is the resulting function. The algorithm is just the tool that shaped it.

This matters because it reframes what you are doing when you train a model. You are not *finding* a pattern that exists in the data. You are *constructing an approximation* of a relationship. The quality of that approximation depends on three things: the quality of your examples, the quality of your representation (features), and the appropriateness of your algorithm for the structure of the problem. The algorithm is only one of three.

### 1.2 The three questions that precede everything

Before touching data, before writing code, before choosing an algorithm, answer these three questions. They constrain every decision that follows.

---

**Question 1: What is the operational reality of this model?**

When this model is deployed and running, what is happening around it? What process is it serving? What decisions does its output feed into? What are the latency, hardware, and reliability requirements of that environment?

Operational reality rules things out. In the Drillbotics geosteering system, operational reality (1 Hz control loop, ≤4 GB RAM, offline during judging, output feeds physical steering decisions) immediately ruled out Transformers, cloud inference, GPU-dependent models, and anything with >100ms latency — not because of theoretical elegance, but because the environment would reject them.

*Write down the operational reality before you start. It will save you from building the wrong thing.*

---

**Question 2: What data is actually available at inference time?**

Not what data exists in the world. Not what data you have in your training set. Specifically: what data will be present, accurate, and timely at the exact moment the model needs to make a prediction in production?

This constraint is almost always more restrictive than people expect, and violating it is one of the most common ways ML projects fail silently. A model that uses data during training that won't exist at inference time will train beautifully and fail completely when deployed. This failure mode is called **training-serving skew** — the gap between the world the model trained in and the world it runs in.

In the TVT geosteering case: the drill bit has a GR sensor producing one reading per foot. At any given moment, you only have what has been drilled so far. Features that look 60 ft *ahead* of the current position would cause training-serving skew — the model would learn from future information that doesn't exist at inference time.

*For every feature you consider, ask: will this exact value be available, computed the same way, at the moment of prediction in production?*

---

**Question 3: What form must the output take to be useful?**

A raw number? A probability? A smoothed curve? A discrete decision (yes/no)? A ranked list? The output format determines how you frame the learning problem, which constrains which algorithms are applicable, which shapes how you build your features.

Output form is often dictated by what consumes the model's predictions. In the TVT case, the output feeds a geosteering controller that makes physical steering decisions. Physical reality is smooth — geology does not jump discontinuously. A raw model output that oscillates ±5 ft between adjacent depth samples would cause the controller to thrash between slide and rotate modes, destroying the drill string. So the output must be smooth. The Savitzky-Golay post-processing step is not optional polish — it is a hard requirement imposed by the downstream system.

*Ask: if the model produced a technically correct prediction in the wrong form, would it still be useful? If not, the form is part of the spec.*

---

### 1.3 The full pipeline

These three questions define the skeleton of every ML system:

```
Operational reality
        │
        ▼
What data exists at inference time?   ← constrains features
        │
        ▼
Feature engineering                   ← translate domain knowledge into numbers
        │
        ▼
Learning algorithm                    ← approximate the function
        │
        ▼
Raw model output
        │
        ▼
Output post-processing                ← shape output to fit what consumes it
        │
        ▼
Downstream system / decision
```

Most ML education covers only the middle row. This handbook covers the whole pipeline.

---

## Part II — Feature Engineering

### 2.1 What feature engineering actually is

Feature engineering is the translation layer between raw data as it exists in the world and the numerical representation a model needs to learn from effectively.

It is not arbitrary manipulation of numbers. It is encoding domain knowledge into a form the algorithm can exploit.

Consider the problem concretely. A LightGBM model sees a table. Each row is an observation, each column is a number. The model has no concept of "this is a time series," "these two columns are measuring the same phenomenon from different angles," or "the relationship between these features changes with depth." It sees numbers and finds splits.

Feature engineering is you, the human, doing the conceptual work before the model sees the data — translating what you know about the domain into numbers the model can act on.

### 2.2 Where feature ideas come from

The most reliable source of feature ideas is this question:

> **What does a domain expert look at when making this judgment manually?**

A geologist steering a horizontal well looks at: the shape of the GR curve (is it sandy or shaly here?), how that shape compares to the typewell reference (are we shallower or deeper in the stack than expected?), how far the offset has drifted since the last survey station, and whether the GR trend is increasing or decreasing (moving toward or away from the target zone).

Each of those looks becomes a feature:
- Shape of the GR curve → rolling statistics (mean, std, range) at multiple window sizes
- Comparison to typewell → cross-correlation best-lag, DTW distance
- Drift from survey station → `dist_from_heel_m`
- GR trend → first and second derivative of GR along depth

The translation from expert judgment to numerical feature is the core skill of feature engineering.

### 2.3 The three categories of features

**Raw signal features** — direct numerical properties of the input signal at the current observation. Useful but limited because they carry no context. Example: `gr` (current GR value), `spp` (standpipe pressure).

**Window / context features** — properties computed over a neighbourhood of the current observation. These encode local structure that a single point cannot express. Window size is a hyperparameter that encodes "how much context is relevant?" — small windows capture local variability, large windows capture regional trends. Example: rolling mean, std, min, max of GR over the last 5, 15, 30, 60 ft.

**Relational features** — properties that express the relationship between two signals, or between the current observation and some reference. These are often the most powerful features because they encode the core structure of the problem. Example: cross-correlation lag between lateral GR and typewell GR, difference between two sensors measuring the same quantity.

In most real problems, raw features alone are weak, window features add useful context, and relational features do the heavy lifting.

### 2.4 The leakage trap

**Data leakage** is when information from outside the model's operational knowledge boundary sneaks into your features or your validation set. It is the most dangerous and most common mistake in applied ML.

There are two forms:

**Feature leakage** — a feature encodes information the model would not have at inference time. Common causes: using a target-derived statistic computed on the full dataset (including the row being predicted), using future values in time-series features, using aggregates computed before train/test split.

**Validation leakage** — your validation set is not independent of your training set, so your error estimates are optimistic. Common causes: random splitting of time-series data (adjacent samples are correlated), random splitting of grouped data (multiple rows per entity), using the full dataset to compute normalisation statistics before splitting.

In the TVT case: if you computed the mean GR per well using all rows (including test rows), and used that as a feature, you would have validation leakage. The fix is to compute all statistics inside the cross-validation loop, using only training-fold data.

*The test for leakage: can the model compute this feature from only the information that will be available at inference time, without any knowledge of the target or of other rows it hasn't seen yet?*

### 2.5 Feature selection intuition

More features are not always better. Each additional feature:
- Adds a dimension to the space the model must explore (curse of dimensionality at extremes)
- Adds noise if the feature is weakly correlated with the target
- Adds training time and inference latency

A rough prioritisation framework:
1. Features that directly encode the mechanism you believe drives the target (relational features from domain knowledge) — include first
2. Features that add context the model would otherwise lack (window features) — include if they improve OOF score
3. Features that are cheap to compute and potentially informative — include tentatively, validate with feature importance
4. Features with high missing rate or low variance — exclude or impute carefully

Feature importance from tree-based models (split count, gain) is a useful diagnostic but not a definitive selector. A feature can have low importance because it is redundant with another feature, not because it is uninformative. Permutation importance is more reliable.

---

## Part III — Validation

### 3.1 Why validation is the hardest part

Training a model is easy. Knowing whether it will work in production is hard. The gap between a model that scores well on a metric and a model that is actually useful is almost always a validation problem.

The purpose of validation is to simulate the model's deployment conditions as faithfully as possible. The closer your validation setup is to the real operational conditions, the more trustworthy your performance estimates.

### 3.2 The cardinal rule: split by the unit of independence

Your validation split must separate data at the level of the entity that is genuinely independent in your problem.

If you are predicting per-row outcomes and rows are independent: random split is fine.

If rows are grouped — multiple rows per patient, per well, per user, per session — rows within a group are correlated. Split by group, not by row. Otherwise adjacent rows from the same group appear in both train and test and the model can interpolate rather than generalise.

If data has a temporal structure — stock prices, sensor readings, drilling data — future data cannot be used to predict the past. Split by time. Never by row.

In the TVT geosteering case: Group K-Fold by `well_id` is mandatory. If you split by row, adjacent depth samples from the same well appear in both folds. The model learns to interpolate between known depths rather than generalise across wells. CV RMSE looks excellent; out-of-competition performance collapses. This is the most common failure mode in geological ML.

### 3.3 The metrics trap

Optimising for the wrong metric is a subtle failure mode. The metric should reflect what matters to the downstream system, not what is easiest to compute.

RMSE penalises large errors heavily (squared term). MAE treats all errors equally. For TVT prediction, where the downstream controller is more sensitive to sudden large jumps than to consistent small offsets, RMSE is the right choice — a single 20 ft prediction error is far more damaging to the geosteering controller than twenty 1 ft errors.

When the downstream system has asymmetric costs (false positives are much worse than false negatives, or vice versa), design your metric accordingly. Do not default to accuracy or RMSE because they are familiar.

---

## Part IV — Output Post-Processing

### 4.1 The model is not the system

A common mistake is to treat the model's raw output as the final product. In almost every real deployment, the raw output requires transformation before it is useful.

Post-processing is not cheating or hiding the model's weaknesses. It is the stage where you enforce constraints that the model cannot learn from data alone — physical constraints, business rules, safety bounds, smoothness requirements.

### 4.2 Types of post-processing

**Smoothing** — when the downstream system requires temporally or spatially smooth outputs. Savitzky-Golay, exponential moving average, Gaussian kernel. Used when abrupt changes in output would cause instability in what consumes it (control systems, UI displays, medical alarms).

**Calibration** — when the model produces probabilities that are systematically overconfident or underconfident. Platt scaling, isotonic regression. Used in risk scoring, medical diagnosis, any application where the *magnitude* of uncertainty matters.

**Clipping / bounding** — when the output must lie within a physically or operationally valid range. A TVT estimate below zero is physically impossible. An ROP prediction above the mechanical limits of the drill string is operationally dangerous. Clip to the valid range.

**Thresholding** — when a continuous score must be converted to a discrete decision. The threshold is a design parameter, not a learned parameter. Set it based on the cost asymmetry of false positives vs. false negatives in the specific deployment context.

**Ensemble averaging** — when multiple models produce independent predictions and averaging reduces variance. The post-processing step combines them with appropriate weights.

### 4.3 Post-processing must be consistent across train and serve

Every transformation applied to model outputs at inference time must be applied consistently. If you smooth predictions in production, smooth OOF predictions in validation too — otherwise your CV metric is measuring the raw output, not the post-processed output that actually runs in production.

This is a subtle form of training-serving skew that is easy to miss and hard to diagnose.

---

## Part V — Putting It Together

### 5.1 The design checklist

Before writing any training code, work through this checklist:

**Operational reality**
- [ ] Where does this model run? What hardware? What latency budget?
- [ ] What process does it serve? What decisions does its output feed?
- [ ] Is it online (real-time) or batch? How often is it called?
- [ ] Are there hard constraints — offline, no GPU, memory cap?

**Data at inference time**
- [ ] List every feature you plan to use. For each one: will this value be available at inference time, computed the same way, without any knowledge of the target?
- [ ] Is there any temporal structure? If yes, are your features backward-looking only?
- [ ] Is data grouped (multiple rows per entity)? If yes, how will you split?

**Feature engineering**
- [ ] What does a domain expert look at when making this judgment manually? Encode each look as a feature.
- [ ] Have you included raw, window, and relational features?
- [ ] Have you checked for the three leakage failure modes: target-derived features, future information, and validation-set contamination?

**Output**
- [ ] What does the downstream system need: a number, a probability, a discrete decision, a curve?
- [ ] Are there physical or operational constraints the output must satisfy?
- [ ] What post-processing is required? Is it applied consistently in both validation and production?

**Validation**
- [ ] What is the unit of independence? Are you splitting at that level?
- [ ] Does your validation simulate the actual deployment conditions?
- [ ] Is your chosen metric aligned with what matters to the downstream system?

### 5.2 The hierarchy of decisions

If you must prioritise, this is the order in which these decisions matter:

1. **Operational reality** — get this wrong and nothing else matters
2. **Inference-time data availability** — get this wrong and your model works in training and fails in production
3. **Validation strategy** — get this wrong and you have no reliable signal about whether anything is working
4. **Feature engineering** — this is where most of the performance comes from in structured data problems
5. **Output post-processing** — often required to make a technically correct model actually safe to deploy
6. **Algorithm choice** — almost always less important than the five decisions above

This ordering will feel counterintuitive if you learned ML from courses that spent 80% of their time on algorithms. It reflects what experienced practitioners actually spend their time on.

### 5.3 A worked example — TVT geosteering

To make the above concrete, here is how every principle in this handbook maps to one real system.

| Principle | Application in TVT geosteering |
|---|---|
| Operational reality | 1 Hz D-WIS loop, ≤4 GB RAM, offline, feeds physical steering decisions → rules out deep models and cloud inference |
| Inference-time data | Only backward-looking GR window available; typewell loaded at startup → no forward-looking features |
| Feature leakage check | Rolling statistics computed per-well in CV loop; no target statistics used |
| Domain expert translation | Geologist looks at GR shape, typewell offset, trend direction → DTW, xcorr lag, GR gradient features |
| Relational features | xcorr best-lag is the single most important feature — it directly encodes the stratigraphic alignment problem |
| Validation strategy | Group K-Fold by well_id — rows within a well are correlated, split must respect this |
| Metric choice | RMSE over MAE — large sudden errors are more damaging to the controller than consistent small ones |
| Output post-processing | Savitzky-Golay smoothing — controller requires smooth TVT curve; raw predictions oscillate |
| Post-processing consistency | SG applied to OOF predictions in CV and to production predictions — same pipeline both places |
| Algorithm choice | LightGBM — edge hardware constraints + small dataset make this the clear choice, not the deep learning default |

---

## Closing note

The mental shift this handbook is asking you to make is from thinking about models as isolated objects to thinking about them as components in systems. A model that scores 0.95 AUC but was trained with leaky features, validated incorrectly, and deployed into an environment where its required data isn't available is not a good model. It is a liability.

The engineers who build reliable ML systems spend most of their time on the edges of the pipeline — on operational constraints, on data availability, on validation integrity, on output form. The algorithm is where they spend the least.

Build the edges first. The middle takes care of itself.
