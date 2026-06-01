# Drillbotics 2026 — Geological State Estimator
## Project Knowledge & Build Plan

> **Team context:** Orcta Technologies / UMaT — Mode Virtual (V), Case 2 (Adaptive Lithology Shift)
> **Core idea:** Train a TVT prediction model on the ROGII Kaggle dataset and deploy it as the geological state estimator inside a closed-loop geosteering controller connected to OpenLab via D-WIS/OPC-UA.

---

## 1. What we are building and why

### The Drillbotics challenge

Drillbotics Mode Virtual (V) requires teams to design, simulate, and control a virtual drilling system using standardised interfaces. We are targeting **Case 2 — Adaptive Lithology Shift**, which requires:

- A drilling controller that adapts to changing rock properties to maximise ROP under constraints
- Realistic handling of slide/rotate modes and dogleg severity limits
- Automatic re-planning based on as-drilled surveys
- Minimum Curvature trajectory output with plan-vs-actual plots
- All AI components must run **offline** on edge hardware (≤2 vCPU, ≤4–8 GB RAM, ≤15 min total inference)

The system connects to the **OpenLab Drilling Simulator** via a **D-WIS/OPC-UA** interface. Setpoints (RPM, WOB, flow rate) are written to OPC-UA nodes; measurements (GR, bit depth, torque, SPP) are read back.

### The geological problem

When a drill bit travels 2–5 km laterally through rock, the geologist must continuously steer the bit to stay within the target productive zone. To make steering decisions, the controller needs to know **where in the stratigraphic column the bit is at every moment** — this is called the **True Vertical Thickness (TVT)** position.

Without a real-time TVT estimate, the controller is blind to geology. It cannot decide whether to slide (steer) or rotate (drill straight), how aggressively to change WOB, or when to re-plan the trajectory.

### Why the ROGII dataset solves this

The **ROGII Wellbore Geology Prediction** Kaggle competition provides:

- Horizontal well GR (Gamma Ray) logs — real-time sensor signal available during drilling
- Typewell reference logs — a vertical well with known full stratigraphy that serves as the geological anchor
- TVT labels — expert geologist interpretations of where the bit was in the stratigraphic column at every 1-ft depth step

This is exactly the data needed to train a geological state estimator. The trained model takes a live GR window plus the typewell reference as input and outputs an estimated TVT position in real time — feeding the Case 2 geosteering controller.

---

## 2. Full system architecture

```
OpenLab Drilling Simulator (plant)
        │
        │  OPC-UA / D-WIS
        ▼
 ┌─────────────────────────────────────────────┐
 │           Drillbotics Controller            │
 │                                             │
 │  ┌──────────────────────────────────────┐   │
 │  │   Signal Discovery Layer             │   │
 │  │   - Browse D-WIS nodes at startup    │   │
 │  │   - Map to model feature names       │   │
 │  │   - Graceful degradation on missing  │   │
 │  └──────────────┬───────────────────────┘   │
 │                 │ live GR + MD stream        │
 │                 ▼                            │
 │  ┌──────────────────────────────────────┐   │
 │  │   Geological State Estimator         │   │  ← trained on ROGII data
 │  │   TVT prediction model (ONNX/pickle) │   │
 │  │   Input:  GR window + typewell ref   │   │
 │  │   Output: estimated TVT position     │   │
 │  └──────────────┬───────────────────────┘   │
 │                 │ TVT estimate               │
 │                 ▼                            │
 │  ┌──────────────────────────────────────┐   │
 │  │   Case 2 Geosteering Controller      │   │
 │  │   - Adaptive ROP optimisation        │   │
 │  │   - Slide / rotate mode decisions    │   │
 │  │   - Dogleg severity enforcement      │   │
 │  │   - Trajectory re-planning           │   │
 │  └──────────────┬───────────────────────┘   │
 │                 │ setpoints                  │
 └─────────────────┼───────────────────────────┘
                   │ OPC-UA writes
                   ▼
         RPM, WOB, Flow rate → back to OpenLab
```

---

## 3. The ROGII dataset and task

### What the data contains

| Source | Description | Role in model |
|---|---|---|
| Horizontal well GR log | Gamma Ray measured along the lateral at ~1 ft intervals | Primary input signal |
| Well trajectory (MD, inclination, azimuth) | Directional survey data | Spatial features |
| Typewell logs | Vertical reference well with known stratigraphy | Geological anchor |
| TVT (target) | Expert-interpreted position in stratigraphic column | Label to predict |

### The core prediction task

**Input:** A window of lateral GR values at the current bit position, the typewell GR reference, and trajectory features.

**Output:** TVT — the interpreted thickness of the geological layer at the current point in true vertical space. This tells the controller "where in the stack is the bit right now?"

### Why this is hard

The horizontal GR and the typewell GR measure the same formations from completely different angles. The horizontal well samples lateral variability; the typewell samples stratigraphic variability. The model must learn to align them continuously as the bit travels through dipping and undulating layers — the alignment offset (lag) changes at every depth step, which is why static correlation fails and sequence-aware approaches are needed.

---

## 4. ML approach

### Model choice: LightGBM

LightGBM was chosen over deep models for three reasons specific to this deployment:

1. **Edge hardware constraint** — the Drillbotics rules cap inference hardware at ≤2 vCPU / ≤4–8 GB RAM. LightGBM runs a 2000-tree ensemble in under 5 ms per well; a Transformer equivalent would be 100–1000× slower.
2. **Small dataset** — most fields have tens to low hundreds of wells. Deep models overfit severely at this scale without heavy regularisation. LightGBM with monotonic constraints and group cross-validation is more reliable.
3. **Interpretability** — feature importances from LightGBM make it easy to verify the model is learning geology (GR-typewell correlation features should dominate) rather than spurious patterns.

### Feature engineering (three groups)

**Group A — Depth and trajectory features**
- `md` — measured depth (primary depth index)
- `md_norm` — depth normalised within well (0→1)
- `dist_from_heel_m` — distance from the start of the lateral
- `gr_grad`, `gr_grad2` — first and second derivative of GR along depth

**Group B — Rolling GR statistics** (computed at windows of 5, 15, 30, 60, 120 ft)
- Rolling mean, std, min, max, range of GR
- GR percentile rank within the well

**Group C — GR–typewell correlation features** (the most important group)
- `xcorr_best_lag` — the typewell depth offset that maximises Pearson correlation between the lateral GR window and the typewell GR. This is the single most important feature — it directly encodes "where in the typewell is the current lateral GR pattern?", which is what TVT is answering.
- `xcorr_best_corr` — the correlation coefficient at the best lag
- `xcorr_mean_corr` — mean correlation across all lags (captures alignment quality)
- `dtw_dist`, `dtw_norm_dist` — Dynamic Time Warping distance between the normalised lateral GR window and the typewell GR window. DTW captures shape similarity independent of amplitude, complementing the correlation lag feature.
- `tw_gr` — typewell GR value at the current interpolated depth
- `gr_tw_diff`, `gr_tw_ratio` — direct lateral vs. typewell GR difference and ratio

### Validation strategy

**Group K-Fold by `well_id`** — this is mandatory, not optional. Adjacent depth samples from the same well are highly correlated; a random train/test split produces fraudulently good CV scores that collapse on the leaderboard. Splitting by geographic area (or well identity) gives honest estimates of generalisation.

Cross-validation is also split by field sub-region where possible to avoid spatial leakage between spatially adjacent wells.

### Physical constraints in the model

Two mechanisms enforce physically plausible TVT predictions:

1. **Monotonic constraint on `md`** — a LightGBM monotone constraint (+1) on the measured depth feature encodes the geological expectation that TVT weakly increases along the lateral. This prevents the model producing oscillating nonsense at depth intervals with ambiguous GR patterns.

2. **Savitzky-Golay post-processing** — a window-31, poly-3 SG filter is applied per well after prediction. TVT should be a smooth curve; abrupt jumps are geologically implausible and would cause unstable geosteering decisions. SG preserves the shape of the curve while removing high-frequency noise.

---

## 5. Model export — pickle and ONNX

The model is exported in two formats for different deployment scenarios.

### Pickle (`tvt_model.pkl`)

Wraps the LightGBM model in a `TVTInferenceModel` dataclass that carries:
- The trained LightGBM model
- The expected feature column names in the correct order
- Smoothing parameters (SG window and polynomial degree)
- Metadata: OOF RMSE, number of training wells

The pickle is the primary runtime for the Python-based Drillbotics controller. It exposes a single `.predict(well_df, typewell_df)` call that handles feature engineering, inference, and smoothing in one step.

### ONNX (`tvt_model.onnx`)

Exported via `skl2onnx` (with `onnxmltools` LightGBM converter). The ONNX model:
- Has no Python or LightGBM dependency at inference time
- Runs via `onnxruntime` which is available on edge hardware with minimal RAM
- Produces float32 TVT predictions directly from a pre-built feature vector
- Can be integrated into non-Python environments (C++, Rust, edge microcontrollers) if needed for future physical mode (Mode P) deployment

The ONNX model requires the feature vector to be pre-built externally (the feature engineering pipeline runs in Python and feeds into ONNX). This is the deployment pattern for the Drillbotics controller: Python builds features from D-WIS signals, ONNX runtime produces the TVT estimate.

---

## 6. Drillbotics runtime integration

### Startup sequence

```python
# 1. Load the frozen TVT estimator
with open('tvt_model.pkl', 'rb') as f:
    tvt_estimator = pickle.load(f)

# 2. Load typewell reference (provided before Phase II)
typewell_df = pd.read_csv('typewell.csv')

# 3. Connect to D-WIS OPC-UA endpoint
# MUST browse and discover nodes — never hard-code node IDs
# (endpoint parameters may change on competition day)
client = OpcUaClient(endpoint)
signals = client.browse_dwis_nodes()   # returns D-WIS named nodes
```

### Per-cycle geological state estimation (1 Hz D-WIS loop)

```python
def estimate_tvt(live_gr_buffer, live_md_buffer, typewell_df):
    """
    live_gr_buffer : last ~120 GR readings from D-WIS
    live_md_buffer : corresponding measured depths
    Returns: TVT estimate at current bit position (float)
    """
    well_df = pd.DataFrame({
        'md':      live_md_buffer,
        'gr':      live_gr_buffer,
        'well_id': 'live',
    })
    preds = tvt_estimator.predict(well_df, typewell_df, smooth=True)
    return float(preds[-1])   # TVT at current bit depth
```

### D-WIS signal mapping

| D-WIS signal | Feature | Role |
|---|---|---|
| `BitDepth` | `md` | Primary depth index |
| `GammaRay` / `GR` | `gr` | Core lithology input |
| `Inclination` | trajectory feature | Trajectory-based features (Case 2) |
| `Azimuth` | trajectory feature | Trajectory-based features (Case 2) |

The signal discovery layer maps whatever D-WIS node names are available on competition day to the model's expected feature names. If a signal is missing, the feature is filled with the training-set median (safe degradation).

### Hardware compliance

| Drillbotics requirement | This model |
|---|---|
| ≤2 vCPU (non-LLM) | <1 vCPU |
| ≤4–8 GB RAM | ~50 MB |
| ≤15 min total inference | <5 ms per well |
| Offline during judging | ✓ no network calls |

---

## 7. Build sequence

The following steps are to be completed in order. Each step has a clear output artifact.

### Step 1 — Train and freeze the TVT model ✅ (in progress)
- **What:** Run `rogii_tvt_pipeline.ipynb` on Kaggle with the competition data
- **Features:** Rolling GR stats + DTW correlation lag + typewell cross-correlation
- **Validation:** Group K-Fold by well ID, OOF RMSE target <10 (top-quartile: <3.0)
- **Output:** `tvt_model.pkl` and `tvt_model.onnx`

### Step 2 — OpenLab connectivity
- **What:** Connect to the OpenLab Drilling Simulator via the D-WIS OPC-UA Docker gateway
- **Verification:** Browse available D-WIS nodes, log a 60-second stream of all signals to CSV
- **Output:** `signal_discovery.py` — reusable discovery module for competition day

### Step 3 — D-WIS adapter and feature bridge
- **What:** Build the layer that maps live D-WIS signal streams to the model's feature vector
- **Design principle:** MUST NOT hard-code node IDs; MUST browse by D-WIS category and match by name
- **Output:** `dwis_adapter.py` — maps discovered nodes → `well_df` format the model expects

### Step 4 — Case 2 geosteering controller
- **What:** The adaptive controller that reads TVT from the estimator and decides RPM/WOB/mode
- **Logic:** Target TVT band → error signal → slide/rotate decision → WOB/RPM setpoint adjustment
- **Output:** `geosteer_controller.py` with state machine (drilling, sliding, rotating, re-planning)

### Step 5 — Minimum Curvature trajectory module
- **What:** Compute and log Minimum Curvature trajectory from survey stations
- **Output:** `trajectory.py` + plan-vs-actual plots (Drillbotics deliverable requirement)

### Step 6 — Integration and closed-loop test
- **What:** Connect all modules end-to-end against OpenLab; run all three Case 2 scenarios
- **Deliverables:** `drilling_timeseries.csv` (≥1 Hz), plan-vs-actual plots, controller decision log

### Step 7 — Edge compliance check
- **What:** Run full inference under the hardware caps (≤2 vCPU, ≤4 GB RAM) and verify <15 min
- **Output:** Benchmark report for Phase II submission

---

## 8. Key risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Competition data columns differ from expected names | Pipeline fails on load | Column normalisation + auto-detection in cell 2 of the notebook |
| Typewell not available in OpenLab | No typewell features — model degrades | GR-only fallback features (Group A+B) still give useful TVT estimates; model retrained without typewell if needed |
| D-WIS node IDs change on competition day | Controller loses signal | Signal discovery layer re-browses at startup; feature bridge uses D-WIS semantic names not node IDs |
| OOF RMSE too high (>15) | Poor geosteering decisions | Add per-well GR amplitude normalisation; increase DTW window; try CatBoost ensemble |
| ONNX export fails | Edge deployment blocked | Pickle path is fully functional; ONNX is secondary for edge but not required for competition judging |
| Small number of training wells | Model overfits | Group K-Fold strictly enforced; monotonic constraint on depth; SG smoothing limits prediction instability |

---

## 9. File inventory

| File | Description |
|---|---|
| `rogii_tvt_pipeline.ipynb` | Full training pipeline — runs on Kaggle |
| `tvt_model.pkl` | Frozen TVT estimator (self-contained, Python) |
| `tvt_model.onnx` | ONNX model for edge runtime |
| `signal_discovery.py` | D-WIS OPC-UA node browser (Step 2) |
| `dwis_adapter.py` | Live signal → feature vector bridge (Step 3) |
| `geosteer_controller.py` | Case 2 adaptive geosteering controller (Step 4) |
| `trajectory.py` | Minimum Curvature trajectory computation (Step 5) |
| `drilling_timeseries.csv` | Competition deliverable — ≥1 Hz logged data |

---

## 10. References

- [Drillbotics Mode Virtual (V) Overview](https://open-source-drilling-community.github.io/drillbotics-guidelines/latest/tracks/group-a/overview/)
- [Drillbotics Technical Specifications](https://open-source-drilling-community.github.io/drillbotics-guidelines/latest/tracks/group-a/technical-specs/)
- [D-WIS Vocabulary Index](https://d-wis.org/vocabulary-index/)
- [OpenLab Drilling Simulator](https://openlab.app/)
- [ROGII Wellbore Geology Prediction — Kaggle](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
- [ROGII competition — GitHub pipeline reference](https://github.com/aaryan2203/rogii-wellbore-geology-prediction-argon)
