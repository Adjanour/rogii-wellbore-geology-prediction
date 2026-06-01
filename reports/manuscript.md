# Wellbore Geology Prediction from Horizontal Well Logs

## A Technical Manuscript on the ROGII Drillbotics Challenge

**May 2026**

---

## Abstract

This manuscript documents the ROGII Wellbore Geology Prediction competition task, its geological foundations, and the machine learning problem it poses. The task requires predicting True Vertical Thickness (TVT) along horizontal wellbores using gamma ray logs measured while drilling, typewell reference data, and wellbore geometry. We provide a self-contained explanation of the well logging and geosteering concepts necessary to understand the dataset, survey the relevant machine learning literature, and present a detailed exploratory analysis of the 773-well training set. The problem reduces to a supervised regression task where the target (TVT) is fully known in training but must be inferred from limited features (MD, X, Y, Z, GR, TVT_input) in the test set, using the typewell as a stratigraphic reference.

---

## 1. Introduction

### 1.1 Competition Context

The ROGII Wellbore Geology Prediction competition was hosted on Kaggle from January to April 2023, sponsored by [ROGII](https://rogii.com/), a company specializing in geosteering and geoscience software (StarSteer). The competition offered a $50,000 prize pool and attracted 319 teams [[1]](#competition-source). Its stated goal: *"Build a model that contributes to automating drilling operations in the oil and gas industry."*

The challenge sits at the intersection of drilling automation, geosteering, and machine learning. Accurate geological prediction along a wellbore is critical for:

- **Optimal well placement** -- keeping the drill bit within the target reservoir zone
- **Formation evaluation** -- identifying lithology boundaries and pay zones
- **Real-time decision making** -- adjusting trajectory based on encountered geology
- **Resource estimation** -- calculating hydrocarbon volumes from accurate formation thicknesses

### 1.2 The Drillbotics Ecosystem

Drillbotics is an international university competition organized by SPE's Drilling Systems Automation Technical Section (DSATS). Teams design drilling rigs (physical or virtual) that autonomously drill rock samples using sensors and control algorithms [[2]](#drillbotics). The ROGII competition extends this theme by applying machine learning to a core geoscience task: predicting subsurface geology from drilling measurements.

---

## 2. Well Logging Fundamentals

This section covers the essential concepts needed to understand the dataset.

### 2.1 Measured Depth vs True Vertical Depth vs True Vertical Thickness

A wellbore is rarely straight. Directional and horizontal wells are intentionally deviated to reach reservoir targets, and even "vertical" wells have small deviations. Three distinct depth measurements arise from this:

**Measured Depth (MD)** is the total length of the wellbore from the surface to a given point, measured along the actual path of the borehole. It corresponds to the length of drill pipe run into the hole [[3]](#aapg-wiki).

**True Vertical Depth (TVD)** is the vertical distance from the surface to a point, measured along a line connecting the point to the center of the earth. For a deviated well, TVD < MD. The correction depends on the inclination angle $\alpha$:

$$TVD = MD \times \cos(\alpha)$$

For a well deviated at 45°, TVD is approximately 70.7% of MD. For a 10° deviation, the difference is ~1.5% [[3]](#aapg-wiki).

**True Vertical Thickness (TVT)** is the thickness of a geological unit measured in the vertical direction. This is distinct from **True Stratigraphic Thickness (TST)** which is measured perpendicular to bedding planes. TVT is the critical measure for volumetric calculations because it is unaffected by variations in bed dip when computed from gridded structural horizons [[3]](#aapg-wiki).

For a deviated well penetrating dipping beds, the TVT calculation requires correcting for both wellbore deviation and formation dip. The general formula is [[4]](#tvt-formula):

$$TVT = MLT \times [\cos\Psi - (\sin\Psi \times \cos\alpha \times \tan\Phi)]$$

Where $\Psi$ = wellbore deviation angle, $\Phi$ = true bed dip, $\alpha$ = acute angle between wellbore azimuth and dip azimuth, and MLT = measured log thickness.

In the dataset, the horizontal well CSVs contain both `TVT` and `TVT_input`. EDA confirms these are **identical** (max absolute difference = 0 in all 773 wells). `TVT_input` is likely the value provided as model input, while both represent the same TVT quantity.

### 2.2 The Gamma Ray (GR) Log

The gamma ray log measures natural radioactivity emitted by formations. Three primary isotopes contribute [[5]](#gr-principles):

- **Potassium-40 (K)** -- emits 1.46 MeV gamma rays; common in illite and K-feldspars
- **Thorium (Th)** -- adsorbed on clay mineral surfaces; common in heavy minerals like monazite
- **Uranium (U)** -- associated with organic matter and phosphatic deposits

**Units**: The API gamma ray unit is defined as 1/200 of the difference between the count rate in a low-radioactivity zone and a high-radioactivity zone in the reference calibration pit at the University of Houston [[5]](#gr-principles). This standard allows consistent comparison across different tools and service companies. A "typical Midcontinent shale" registers approximately 100 API units.

**Interpretation**: The GR log is primarily used as a *shale log*. Shales concentrate radioactive isotopes and therefore produce high GR readings. Clean sandstones, limestones, and dolomites produce low readings [[6]](#gr-usage). The typical cutoff separating sand from shale is around 60 API units, but this varies by basin.

In the dataset, GR values range from 14 to 487 API units (mean 88, std 24). Notably, approximately 43% of horizontal well rows have NaN GR, indicating the log was not acquired at every depth point along the measured depth trajectory.

### 2.3 Typewells and Horizontal Well Correlation

A **typewell** (also called a **type log** or **offset well**) serves as a stratigraphic reference. It is typically a vertical well in the same field that has been comprehensively logged and often cored. Its gamma ray signature provides a "fingerprint" of the vertical stratigraphic section [[7]](#typewell).

**Geosteering** -- the process of steering a horizontal wellbore within a target reservoir zone -- relies on comparing logging data from the horizontal well to the type log. As described by Koury: *"The basis of geosteering is correlating to marker beds that have lateral continuity"* [[8]](#geosteering). During drilling, the geologist compares the real-time MWD gamma ray from the horizontal well to the type log to determine the wellbore's stratigraphic position.

Key challenges in this correlation include:
- **Stretching and squeezing** -- horizontal wells traverse formations at oblique angles, so the log response must be vertically compressed or expanded to match the type log [[9]](#correlation-window)
- **Lateral variability** -- formation thickness and gamma character can change significantly over distances of hundreds of meters [[8]](#geosteering)
- **Measurement differences** -- MWD gamma tools (used in horizontal drilling) differ from wireline gamma tools (used in vertical typewells), introducing systematic offsets

In this dataset, each well ID has both a `__horizontal_well.csv` (logging data at MD increments) and a `__typewell.csv` (TVT-GR pairs, often with Geology labels). The typewell represents the expected vertical geology at the surface location of the horizontal well's landing point.

### 2.4 Formation Tops and Geology Labels

A **formation top** is the boundary (contact) between two geologic formations, identified by a distinctive change in log response. Geologists manually "pick" these tops during well log interpretation [[10]](#predictatops).

The typewells contain 11 geology classes organized in descending stratigraphic order:

| Label | Relative Depth | Description |
|---|---|---|
| ANCC | Shallowest | Likely Austin Chalk or equivalent carbonate |
| ASTNU | ↑ | Upper member of a formation unit |
| ASTNL | ↑ | Lower member |
| EGFDU | ↑ | Upper Eagle Ford equivalent |
| EGFDL | ↑ | Lower Eagle Ford equivalent |
| LTHL | ↑ | A lithologic unit |
| LTGT | ↑ | A lithologic unit |
| LBHL | ↑ | A lithologic unit |
| MNSS | ↑ | A lithologic unit |
| BUDA | Deepest | Buda Limestone |

These span from shallow carbonate (ANCC) through shale-rich intervals (Eagle Ford equivalents) to the deeper Buda limestone. The labels form contiguous TVT intervals in the typewell, with the specific depth range varying by well (TVT ranges differ by up to 3000 ft across wells for the same formation, reflecting different structural positions).

---

## 3. Dataset Structure and Exploratory Analysis

### 3.1 Dataset Composition

| Split | Number of Wells | Horizontal Well Columns | Typewell Columns |
|---|---|---|---|
| Train | 773 | 13 (MD, X, Y, Z, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA, TVT, GR, TVT_input) | 3 (TVT, GR, Geology) |
| Test | 3 | 6 (MD, X, Y, Z, GR, TVT_input) | 2 (TVT, GR) |

### 3.2 Training Data Statistics

| Variable | Mean | Std | Min | Max |
|---|---|---|---|---|
| Horizontal well rows/well | 6,588 | -- | 2,058 | 12,141 |
| Typewell rows/well | 2,027 | -- | 636 | 10,043 |
| GR (API) | 87.8 | 23.8 | 13.9 | 487.0 |
| TVT / TVT_input (ft) | ~11,500 | ~650 | 9,245 | 12,894 |
| MD length/well (ft) | 6,587 | -- | 2,057 | 12,140 |
| Well X-range (ft) | 2,938 | -- | 20 | 7,541 |
| Well Y-range (ft) | 5,131 | -- | 318 | 10,206 |
| Well Z-range (ft) | 788 | -- | 198 | 1,412 |

### 3.3 Geology Label Distribution

Across all typewells, 66.6% of rows have a Geology label. The distribution is highly imbalanced:

- **ANCC** is the most common (28.2% of labeled rows)
- **EGFDL** (19.7%), **ASTNL** (16.5%), **BUDA** (13.5%), **ASTNU** (11.3%) are moderately represented
- **EGFDU** (6.7%) and **MNSS** (0.5%) are less common
- **LTHL** (0.11%), **LBHL** (0.16%), and **LTGT** (0.09%) are extremely rare

This has implications for any classification-based approach: rare classes will require special handling (class weighting, oversampling, or a two-stage prediction strategy).

### 3.4 Training vs Test Discrepancy

The test set is missing several columns present in training:

- **No ANCC/ASTNU/.../BUDA formation top depths** -- these six columns are absent from test horizontal wells
- **No TVT target** -- this is what must be predicted
- **No Geology column** in the typewell

This means the model cannot rely on knowing formation top depths at inference time. The available features for test prediction are: MD, X, Y, Z, GR, TVT_input, plus the typewell's TVT-GR relationship.

### 3.5 TVT and TVT_input

TVT and TVT_input are identical across all 773 training wells. Their role in the problem appears to be:
- **TVT** is the regression target (what we predict)
- **TVT_input** is the *same value* provided as a feature (likely a positional interpolant, e.g., converting MD to TVT using the well survey and assuming horizontal bedding)

Since TVT_input equals TVT in training but will differ in test, the model must learn to use it as a proxy rather than a shortcut.

---

## 4. Machine Learning Problem Formulation

### 4.1 Task Definition

The problem is **supervised time series regression**: for each depth increment along the test horizontal wells, predict a continuous TVT value.

Formally: Given a horizontal well $W$ with $n$ depth points $\{p_1, p_2, ..., p_n\}$, where each $p_i$ has features $\mathbf{x}_i = (MD_i, X_i, Y_i, Z_i, GR_i, TVT\_input_i)$, and given a typewell $T$ with $\{TVT_j, GR_j\}_{j=1}^m$, predict $\hat{y}_i = TVT_i$ for each test point.

### 4.2 Why This Is Challenging

1. **Missing GR in training** -- 43% of horizontal well GR values are NaN, requiring imputation
2. **Training-test asymmetry** -- formation top columns present only in training mean the learned mapping must be robust to their absence
3. **Rare geology labels** -- LTHL, LTGT, LBHL have fewer than 2000 labeled rows across 773 wells
4. **Variable well geometries** -- MD length varies 6-fold; XY ranges vary 300-fold; each well has unique geometry
5. **GR measurement mismatch** -- training horizontal GR (likely MWD gamma) and typewell GR (likely wireline gamma) are from different tool types with different responses

### 4.3 Related Work in ML for Well Log Analysis

**Well log correlation** has been approached with dynamic time warping (DTW) algorithms that identify similar log segments between pairs of wells [[11]](#auto-correlation). Supervised approaches using random forests, gradient boosting, and neural networks have been applied to lithofacies classification from GR, resistivity, density, and neutron porosity logs, achieving 92-97% accuracy in basin-specific studies [[12]](#ml-lithology).

**Recommender system approaches** use matrix factorization on formation top pick data alone (no well logs required) to predict missing tops [[13]](#recommender). However, these require dense top coverage across many wells.

**Deep learning approaches** have recently combined CNNs and transformers for stratigraphic correlation, achieving F1 scores of 0.89 on blind tests by incorporating geological constraints into the loss function [[14]](#deep-learning). These methods require substantial training data but can capture multi-scale patterns.

For this specific task, a sequence-to-sequence architecture that consumes GR-TVT_input windows and produces TVT estimates -- conditioned on a learned representation of the typewell -- would be a natural fit. Simpler baselines include gradient-boosted trees on windowed features (lagged MD, GR, spatial gradients) with typewell information injected as a similarity-weighted lookup.

---

## 5. Conclusion

The ROGII Wellbore Geology Prediction task is a well-posed machine learning problem grounded in decades of petrophysical and geosteering practice. The dataset captures a realistic scenario: predict a continuous geology property (TVT) at each depth point along a horizontal well, using the gamma ray log and a vertical typewell as reference. Key findings from EDA include the TVT/TVT_input identity, the 43% GR missingness pattern, the extreme class imbalance in geology labels, and the training-test feature discrepancy. A successful solution will need to handle these challenges while leveraging the typewell's stratigraphic information.

---

## References

<a id="competition-source"></a>[1] ROGII -- Wellbore Geology Prediction. Kaggle, 2023. Summarized at https://www.competehub.dev/en/competitions/kagglerogii-wellbore-geology-prediction

<a id="drillbotics"></a>[2] Drillbotics International Competition. SPE DSATS. https://drillbotics.com/about-drillbotics/

<a id="aapg-wiki"></a>[3] Depth and thickness conversion. AAPG Wiki. https://wiki.aapg.org/Depth_and_thickness_conversion

<a id="tvt-formula"></a>[4] Thickness Determinations for Volumetric Calculations. https://geojager.tripod.com/Volumetric_Calculations.pdf

<a id="gr-principles"></a>[5] Gamma Ray Log Principles. In: Petrophysics by Paul Glover. https://homepages.see.leeds.ac.uk/~earpwjg/PG_EN/CD%20Contents/GGL-66565%20Petrophysics%20English/Chapter%2011.PDF

<a id="gr-usage"></a>[6] Reading the Rocks from Wireline Logs -- Gamma Ray Log. Kansas Geological Survey. https://www.kgs.ku.edu/PRS/ReadRocks/GRLog.html

<a id="typewell"></a>[7] Type-log selection and preparation for a more accurate geosteering model. Search and Discovery, 2015. Laubhan, A.

<a id="geosteering"></a>[8] Koury, C. Geosteering in the Wolfcamp Formation -- Challenges and Solutions. LinkedIn / AAPG, 2021. https://www.linkedin.com/pulse/geosteering-wolfcamp-formation-challenges-solutions-chad-koury

<a id="correlation-window"></a>[9] Correlation Window -- Horizontal vs Vertical Well Correlation. IHS Petra Help. https://onlinehelp.ihs.com/Energy/Petra/2022/Content/11-DirWellMod/dwm_correlate.htm

<a id="predictatops"></a>[10] Gosses, J. Stratigraphic pick prediction via supervised machine-learning: Predictatops. 2019. http://justingosses.com/blog/predictatops

<a id="auto-correlation"></a>[11] Automatic Well Log Correlation. Search and Discovery, 2017. Abstract 90291ACE.

<a id="ml-lithology"></a>[12] Machine learning techniques for lithology prediction using wireline logs. *Scientific Reports*, 2025. https://doi.org/10.1038/s41598-025-18670-y

<a id="recommender"></a>[13] Well-log correlation using collaborative filtering. arXiv:2202.08869, 2022. https://export.arxiv.org/pdf/2202.08869v1.pdf

<a id="deep-learning"></a>[14] Stratigraphic Correlation of Well Logs Using Geology-Informed Deep Learning Networks. *Processes*, 2025. https://doi.org/10.3390/pr13051288
