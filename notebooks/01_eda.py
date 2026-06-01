"""
Exploratory Data Analysis for Wellbore Geology Prediction
Run: python notebooks/01_eda.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data.loader import (
    load_all_wells,
    load_horizontal,
    load_typewell,
    GEOLOGY_ORDER,
)
from src.visualization.visualize import plot_well_logs, plot_formation_tops

DATA_DIR = Path(__file__).resolve().parent.parent / "train"
REPORTS = Path(__file__).resolve().parent.parent / "reports" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("WELLBORE GEOLOGY PREDICTION - EDA")
print("=" * 60)

# --- 1. Load all training wells ---
print("\n[1] Loading training data...")
wells = load_all_wells(DATA_DIR.parent, is_train=True)
print(f"    Total wells: {len(wells)}")

# --- 2. Dataset statistics ---
print("\n[2] Dataset statistics:")
hw_lengths = []
tw_lengths = []
n_formations = []
for wid, (hw, tw) in wells.items():
    hw_lengths.append(len(hw))
    tw_lengths.append(len(tw))
    n_formations.append(hw.filter(GEOLOGY_ORDER).nunique().sum())

hw_lengths = np.array(hw_lengths)
tw_lengths = np.array(tw_lengths)
print(
    f"    Horizontal well rows: min={hw_lengths.min()}, max={hw_lengths.max()}, "
    f"mean={hw_lengths.mean():.0f}, median={np.median(hw_lengths):.0f}"
)
print(
    f"    Typewell rows:        min={tw_lengths.min()}, max={tw_lengths.max()}, "
    f"mean={tw_lengths.mean():.0f}, median={np.median(tw_lengths):.0f}"
)

# --- 3. Geology label distribution ---
print("\n[3] Geology label distribution across all typewells:")
all_geo = pd.concat([tw["Geology"] for _, (_, tw) in wells.items()])
geo_counts = all_geo.value_counts(dropna=False)
total_labeled = all_geo.dropna().shape[0]
print(
    f"    Total rows: {len(all_geo)}, Labeled: {total_labeled} ({100 * total_labeled / len(all_geo):.1f}%)"
)
print(f"    Unlabeled: {geo_counts.get(np.nan, 0)}")
for g in GEOLOGY_ORDER:
    cnt = geo_counts.get(g, 0)
    pct = 100 * cnt / max(total_labeled, 1)
    print(f"      {g:8s}: {cnt:6d} ({pct:5.2f}%)")

fig, ax = plt.subplots(figsize=(10, 4))
geo_counts_dropna = geo_counts.dropna()
colors = plt.cm.tab10(np.linspace(0, 1, len(geo_counts_dropna)))
bars = ax.bar(range(len(geo_counts_dropna)), geo_counts_dropna.values, color=colors)
ax.set_xticks(range(len(geo_counts_dropna)))
ax.set_xticklabels(geo_counts_dropna.index, rotation=45, ha="right")
ax.set_ylabel("Count")
ax.set_title("Geology Label Distribution (all typewells)")
for bar, val in zip(bars, geo_counts_dropna.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 20, str(val), ha="center", fontsize=7
    )
plt.tight_layout()
fig.savefig(REPORTS / "geology_distribution.png", bbox_inches="tight")
plt.close()
print(f"    Saved: reports/figures/geology_distribution.png")

# --- 4. Check for missing values in training data ---
print("\n[4] Missing value check (sample well):")
hw_sample, tw_sample = wells[list(wells.keys())[0]]
print("    Horizontal well missing:")
print(f"      {(hw_sample.isna().sum() / len(hw_sample) * 100).to_dict()}")
print("    Typewell missing:")
print(f"      {(tw_sample.isna().sum() / len(tw_sample) * 100).to_dict()}")

# --- 5. GR and TVT distributions ---
print("\n[5] Feature distributions (across all wells, sampled):")
all_gr = pd.concat([hw["GR"] for hw, _ in wells.values()])
all_tvt = pd.concat([hw["TVT"] for hw, _ in wells.values()])
all_tvt_input = pd.concat([hw["TVT_input"] for hw, _ in wells.values()])
print(
    f"    GR:        mean={all_gr.mean():.1f}, std={all_gr.std():.1f}, "
    f"min={all_gr.min():.1f}, max={all_gr.max():.1f}"
)
print(
    f"    TVT:       mean={all_tvt.mean():.1f}, std={all_tvt.std():.1f}, "
    f"min={all_tvt.min():.1f}, max={all_tvt.max():.1f}"
)
print(
    f"    TVT_input: mean={all_tvt_input.mean():.1f}, std={all_tvt_input.std():.1f}, "
    f"min={all_tvt_input.min():.1f}, max={all_tvt_input.max():.1f}"
)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, data, title, color, xlabel in zip(
    axes,
    [all_gr, all_tvt, all_tvt_input],
    ["Gamma Ray (GR)", "TVT", "TVT_input"],
    ["green", "blue", "orange"],
    ["GR (API)", "TVT (ft)", "TVT_input (ft)"],
):
    ax.hist(data.sample(min(50000, len(data))), bins=80, color=color, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
plt.tight_layout()
fig.savefig(REPORTS / "feature_distributions.png", bbox_inches="tight")
plt.close()
print(f"    Saved: reports/figures/feature_distributions.png")

# --- 6. TVT vs TVT_input ---
print("\n[6] TVT vs TVT_input check:")
diff = pd.concat([(hw["TVT"] - hw["TVT_input"]).abs() for hw, _ in wells.values()])
print(f"    Max absolute difference: {diff.max():.6f}")
print(f"    Are they identical? {diff.max() < 1e-10}")

# --- 7. Plot sample wells ---
print("\n[7] Generating well log plots for first 6 wells...")
sample_ids = list(wells.keys())[:6]
for wid in sample_ids:
    hw, tw = wells[wid]
    plot_well_logs(hw, tw, wid, save_to=REPORTS / f"well_{wid}_logs.png")
    if "ANCC" in hw.columns:
        plot_formation_tops(hw, wid, save_to=REPORTS / f"well_{wid}_tops.png")
print(f"    Saved to reports/figures/")

# --- 8. Geology depth intervals (across wells) ---
print("\n[8] Geology interval TVT ranges (pooled across wells):")
geo_tvt = pd.concat(
    [tw[["TVT", "Geology"]].dropna().assign(well=wid) for wid, (_, tw) in wells.items()]
)
for g in GEOLOGY_ORDER:
    grp = geo_tvt[geo_tvt["Geology"] == g]["TVT"]
    print(
        f"    {g:8s}: TVT = [{grp.min():10.1f}, {grp.max():10.1f}], "
        f"span = {grp.max() - grp.min():.1f} ft"
    )

# --- 9. Well path statistics ---
print("\n[9] Well path geometry:")
hw_all = pd.concat([hw.assign(well=wid) for wid, (hw, _) in wells.items()])
well_stats = (
    hw_all.groupby("well")
    .agg(
        MD_start=("MD", "min"),
        MD_end=("MD", "max"),
        MD_length=("MD", lambda x: x.max() - x.min()),
        X_range=("X", lambda x: x.max() - x.min()),
        Y_range=("Y", lambda x: x.max() - x.min()),
        Z_range=("Z", lambda x: x.max() - x.min()),
    )
    .reset_index()
)
print(
    f"    MD length:   mean={well_stats['MD_length'].mean():.0f}, "
    f"min={well_stats['MD_length'].min():.0f}, max={well_stats['MD_length'].max():.0f}"
)
print(
    f"    X range:     mean={well_stats['X_range'].mean():.0f}, "
    f"min={well_stats['X_range'].min():.0f}, max={well_stats['X_range'].max():.0f}"
)
print(
    f"    Y range:     mean={well_stats['Y_range'].mean():.0f}, "
    f"min={well_stats['Y_range'].min():.0f}, max={well_stats['Y_range'].max():.0f}"
)
print(
    f"    Z range:     mean={well_stats['Z_range'].mean():.0f}, "
    f"min={well_stats['Z_range'].min():.0f}, max={well_stats['Z_range'].max():.0f}"
)

# --- 10. Test set ---
print("\n[10] Test set overview:")
test_wells = load_all_wells(DATA_DIR.parent, is_train=False)
for wid, (hw, tw) in test_wells.items():
    print(f"    {wid}: horizontal={len(hw)} rows, typewell={len(tw)} rows")
    print(f"      Horizontal cols: {list(hw.columns)}")
    print(f"      Typewell cols:   {list(tw.columns)}")
    print(f"      TVT_input range: [{hw['TVT_input'].min():.1f}, {hw['TVT_input'].max():.1f}]")

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
