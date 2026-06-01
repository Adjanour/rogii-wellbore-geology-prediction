import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def plot_well_logs(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    well_id: str,
    save_to: Path | None = None,
):
    fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=False)
    fig.suptitle(f"Well {well_id}", fontsize=13, fontweight="bold")

    md = hw["MD"]

    # 1. GR log
    ax = axes[0]
    ax.plot(hw["GR"], md, color="green", linewidth=0.6)
    ax.set_xlabel("GR (API)")
    ax.set_ylabel("Measured Depth (ft)")
    ax.invert_yaxis()
    ax.set_title("Gamma Ray")

    # 2. TVT profile
    ax = axes[1]
    ax.plot(hw["TVT"] if "TVT" in hw.columns else hw["TVT_input"], md, color="blue", linewidth=0.6)
    ax.set_xlabel("TVT (ft)")
    ax.invert_yaxis()
    ax.set_title("True Vertical Thickness")

    # 3. Wellbore trajectory (X-Y)
    ax = axes[2]
    ax.plot(hw["X"], hw["Y"], color="purple", linewidth=0.5)
    ax.scatter(hw["X"].iloc[0], hw["Y"].iloc[0], color="green", s=30, label="Start")
    ax.scatter(hw["X"].iloc[-1], hw["Y"].iloc[-1], color="red", s=30, label="End")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Well Path")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # 4. Typewell GR + Geology
    ax = axes[3]
    ax.plot(tw["GR"], tw["TVT"], color="black", linewidth=0.6)
    ax.set_xlabel("GR (API)")
    ax.set_ylabel("TVT (ft)")
    ax.invert_yaxis()
    ax.set_title("Typewell")

    if "Geology" in tw.columns:
        tw_labeled = tw.dropna(subset=["Geology"])
        if len(tw_labeled):
            colors = plt.cm.tab10(np.linspace(0, 1, tw_labeled["Geology"].nunique()))
            cdict = dict(zip(tw_labeled["Geology"].unique(), colors))
            for g, grp in tw_labeled.groupby("Geology"):
                mid = grp["TVT"].mean()
                ax.annotate(
                    g, xy=(grp["GR"].max() + 5, mid), fontsize=6, color=cdict[g], va="center"
                )
                ax.axhspan(grp["TVT"].min(), grp["TVT"].max(), color=cdict[g], alpha=0.15)

    plt.tight_layout()
    if save_to:
        fig.savefig(save_to, bbox_inches="tight")
    plt.close(fig)


def plot_formation_tops(hw: pd.DataFrame, well_id: str, save_to: Path | None = None):
    top_cols = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
    fig, ax = plt.subplots(figsize=(10, 6))
    md = hw["MD"]
    for col in top_cols:
        ax.plot(md, hw[col], label=col, linewidth=0.8)
    ax.set_xlabel("Measured Depth (ft)")
    ax.set_ylabel("Depth (ft)")
    ax.set_title(f"Well {well_id} - Formation Tops")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    if save_to:
        fig.savefig(save_to, bbox_inches="tight")
    plt.close(fig)
