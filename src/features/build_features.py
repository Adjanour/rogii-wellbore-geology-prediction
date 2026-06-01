import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist


def impute_col(df: pd.DataFrame, col: str, key: str = "MD") -> pd.DataFrame:
    df = df.copy()
    m = df[col].isna()
    if m.any():
        known = df.loc[~m, key].values
        known_v = df.loc[~m, col].values
        if len(known) >= 2:
            f = interp1d(known, known_v, kind="linear", fill_value="extrapolate")
            df.loc[m, col] = f(df.loc[m, key].values)
        else:
            df.loc[m, col] = df[col].mean()
    return df


def impute_gr(hw: pd.DataFrame) -> pd.DataFrame:
    return impute_col(hw, "GR")


def impute_tvt_input(hw: pd.DataFrame) -> pd.DataFrame:
    hw["TVT_input_missing"] = hw["TVT_input"].isna().astype(np.float32)
    return impute_col(hw, "TVT_input")


def compute_well_geometry(hw: pd.DataFrame) -> pd.DataFrame:
    hw = hw.copy()
    dx = hw["X"].diff()
    dy = hw["Y"].diff()
    dz = hw["Z"].diff()
    dmd = hw["MD"].diff().clip(lower=1e-6)

    inc = np.arctan2(np.sqrt(dx**2 + dy**2), -dz)
    hw["inclination_rad"] = inc
    hw["inclination_deg"] = np.degrees(inc)

    azi = np.arctan2(dx, dy)
    hw["azimuth_rad"] = azi
    hw["azimuth_deg"] = np.degrees(azi) % 360

    dls = (
        np.degrees(
            np.arccos(
                np.clip(
                    np.cos(inc.shift()) * np.cos(inc)
                    + np.sin(inc.shift()) * np.sin(inc) * np.cos(azi.shift() - azi),
                    -1,
                    1,
                )
            )
        )
        / dmd
        * 100
    )
    hw["dls"] = dls.fillna(0)

    hw["curvature"] = dls / dmd.replace(0, np.nan)
    hw["curvature"] = hw["curvature"].fillna(0)

    return hw


def build_typewell_features(
    hw: pd.DataFrame, tw: pd.DataFrame, method: str = "nearest"
) -> pd.DataFrame:
    hw = hw.copy()
    hw_gr = hw["GR"].values
    tw_gr = tw["GR"].values
    tw_tvt = tw["TVT"].values

    if method == "nearest":
        from scipy.spatial import KDTree

        tree = KDTree(tw_gr.reshape(-1, 1))
        best_dist, best_idx = tree.query(hw_gr.reshape(-1, 1), k=1)
        best_idx = best_idx.ravel()
        best_dist = best_dist.ravel()
    else:
        dist = cdist(hw_gr.reshape(-1, 1), tw_gr.reshape(-1, 1), metric="euclidean")
        best_idx = dist.argmin(axis=1)
        best_dist = dist.min(axis=1)

    hw["tw_tvt_match"] = tw_tvt[best_idx]
    hw["tw_gr_match"] = tw_gr[best_idx]
    hw["tw_gr_dist"] = best_dist
    hw["tw_gr_sim"] = np.exp(-best_dist / (best_dist.std() + 1e-8))

    return hw


def add_window_features(
    hw: pd.DataFrame, col: str = "GR", windows: list | None = None
) -> pd.DataFrame:
    hw = hw.copy()
    if windows is None:
        windows = [5, 11, 21, 51]
    for w in windows:
        hw[f"{col}_mean_{w}"] = hw[col].rolling(w, center=True, min_periods=1).mean()
        hw[f"{col}_std_{w}"] = hw[col].rolling(w, center=True, min_periods=1).std().fillna(0)
        hw[f"{col}_min_{w}"] = hw[col].rolling(w, center=True, min_periods=1).min()
        hw[f"{col}_max_{w}"] = hw[col].rolling(w, center=True, min_periods=1).max()
        p = hw[col].rolling(w, center=True, min_periods=1)
        hw[f"{col}_p25_{w}"] = p.quantile(0.25)
        hw[f"{col}_p75_{w}"] = p.quantile(0.75)
    return hw


def build_features(hw: pd.DataFrame, tw: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    hw = impute_gr(hw)
    hw = impute_tvt_input(hw)
    hw = compute_well_geometry(hw)
    hw = build_typewell_features(hw, tw)
    hw = add_window_features(hw)

    geom_cols = ["inclination_deg", "azimuth_deg", "dls", "curvature"]
    tw_cols = ["tw_tvt_match", "tw_gr_match", "tw_gr_dist", "tw_gr_sim"]
    window_cols = [c for c in hw.columns if c.startswith("GR_")]

    base_features = ["MD", "X", "Y", "Z", "GR", "TVT_input", "TVT_input_missing"]
    feature_cols = base_features + geom_cols + tw_cols + window_cols

    keep_cols = [c for c in feature_cols if c in hw.columns]
    result = hw[keep_cols].copy()

    if is_train and "TVT" in hw.columns:
        result["TVT"] = hw["TVT"].values

    return result


FEATURE_COLS = [
    "MD",
    "X",
    "Y",
    "Z",
    "GR",
    "TVT_input",
    "TVT_input_missing",
    "inclination_deg",
    "azimuth_deg",
    "dls",
    "curvature",
    "tw_tvt_match",
    "tw_gr_match",
    "tw_gr_dist",
    "tw_gr_sim",
    "GR_mean_5",
    "GR_std_5",
    "GR_min_5",
    "GR_max_5",
    "GR_p25_5",
    "GR_p75_5",
    "GR_mean_11",
    "GR_std_11",
    "GR_min_11",
    "GR_max_11",
    "GR_p25_11",
    "GR_p75_11",
    "GR_mean_21",
    "GR_std_21",
    "GR_min_21",
    "GR_max_21",
    "GR_p25_21",
    "GR_p75_21",
    "GR_mean_51",
    "GR_std_51",
    "GR_min_51",
    "GR_max_51",
    "GR_p25_51",
    "GR_p75_51",
]
