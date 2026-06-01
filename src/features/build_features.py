import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from dtaidistance import dtw

WINDOWS = [5, 15, 30, 60, 120]

CORR_LAGS = list(range(-60, 61, 5))
DTW_WINDOW = 30


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


def interpolate_typewell_gr(typewell_df, lateral_md: np.ndarray) -> np.ndarray:
    if typewell_df is None or typewell_df.empty:
        return np.full(len(lateral_md), np.nan)
    tw_depth_col = next(
        (c for c in typewell_df.columns if c in ("TVT", "depth", "tvd", "md", "depth_m")), None
    )
    tw_gr_col = next((c for c in typewell_df.columns if c in ("GR", "gamma_ray", "gr_api")), None)
    if tw_depth_col is None or tw_gr_col is None:
        return np.full(len(lateral_md), np.nan)
    tw = typewell_df[[tw_depth_col, tw_gr_col]].dropna().sort_values(tw_depth_col)
    if len(tw) < 2:
        return np.full(len(lateral_md), np.nan)
    f = interp1d(
        tw[tw_depth_col].values,
        tw[tw_gr_col].values,
        bounds_error=False,
        fill_value=(tw[tw_gr_col].iloc[0], tw[tw_gr_col].iloc[-1]),
    )
    return f(lateral_md)


def rolling_gr_features(gr: np.ndarray, md: np.ndarray) -> pd.DataFrame:
    s = pd.Series(gr)
    out = {"md": md, "md_norm": (md - md.min()) / (md.max() - md.min() + 1e-9)}

    out["gr"] = gr
    out["gr_grad"] = np.gradient(gr, md)
    out["gr_grad2"] = np.gradient(out["gr_grad"], md)
    out["gr_pct_rank"] = s.rank(pct=True).values

    for w in WINDOWS:
        r = s.rolling(w, center=True, min_periods=1)
        out[f"gr_mean_{w}"] = r.mean().values
        out[f"gr_std_{w}"] = r.std().fillna(0).values
        out[f"gr_min_{w}"] = r.min().values
        out[f"gr_max_{w}"] = r.max().values
        out[f"gr_range_{w}"] = out[f"gr_max_{w}"] - out[f"gr_min_{w}"]

    return pd.DataFrame(out)


def xcorr_best_lag(
    lateral_gr: np.ndarray,
    tw_gr: np.ndarray,
    center: int,
    half_win: int = DTW_WINDOW,
    lags: list = CORR_LAGS,
) -> dict:
    lo = max(0, center - half_win)
    hi = min(len(lateral_gr), center + half_win)
    win = lateral_gr[lo:hi]
    if len(win) < 5:
        return {"xcorr_best_lag": 0.0, "xcorr_best_corr": 0.0, "xcorr_mean_corr": 0.0}
    corrs = []
    for lag in lags:
        tlo = max(0, lo + lag)
        thi = min(len(tw_gr), hi + lag)
        if thi - tlo != len(win):
            corrs.append(0.0)
            continue
        tw_win = tw_gr[tlo:thi]
        if tw_win.std() < 1e-6 or win.std() < 1e-6:
            corrs.append(0.0)
        else:
            corrs.append(np.corrcoef(win, tw_win)[0, 1])
    best_idx = int(np.argmax(corrs))
    return {
        "xcorr_best_lag": float(lags[best_idx]),
        "xcorr_best_corr": float(corrs[best_idx]),
        "xcorr_mean_corr": float(np.mean(corrs)),
    }


def compute_dtw(
    lateral_gr: np.ndarray, tw_gr: np.ndarray, center: int, half_win: int = DTW_WINDOW
) -> dict:
    lo = max(0, center - half_win)
    hi = min(len(lateral_gr), center + half_win)
    win_lat = lateral_gr[lo:hi].astype(np.float64)
    win_tw = tw_gr[lo:hi].astype(np.float64)
    if len(win_lat) < 5 or np.isnan(win_tw).any():
        return {"dtw_dist": np.nan, "dtw_norm_dist": np.nan}

    def norm01(x):
        r = x.max() - x.min()
        return (x - x.min()) / (r + 1e-9)

    d = dtw.distance_fast(norm01(win_lat), norm01(win_tw), window=half_win // 2)
    return {"dtw_dist": float(d), "dtw_norm_dist": float(d / len(win_lat))}


def build_features(hw: pd.DataFrame, tw: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    hw = hw.sort_values("MD").copy()
    md = hw["MD"].values
    gr = impute_gr(hw)["GR"].values

    feat = rolling_gr_features(gr, md)

    tw_gr = interpolate_typewell_gr(tw, md)
    xcorr_rows = [xcorr_best_lag(gr, tw_gr, i) for i in range(len(md))]
    dtw_rows = [compute_dtw(gr, tw_gr, i) for i in range(len(md))]

    feat = pd.concat(
        [
            feat,
            pd.DataFrame(xcorr_rows),
            pd.DataFrame(dtw_rows),
        ],
        axis=1,
    )

    feat["tw_gr"] = tw_gr
    feat["gr_tw_diff"] = gr - tw_gr
    feat["gr_tw_ratio"] = gr / (tw_gr + 1e-9)

    if is_train:
        hw = impute_tvt_input(hw)
        hw = compute_well_geometry(hw)
        feat["TVT_input"] = hw["TVT_input"].values
        feat["TVT_input_missing"] = hw["TVT_input_missing"].values
        feat["inclination_deg"] = hw["inclination_deg"].values
        feat["azimuth_deg"] = hw["azimuth_deg"].values
        feat["dls"] = hw["dls"].values
        feat["curvature"] = hw["curvature"].values

    if is_train and "TVT" in hw.columns:
        feat["TVT"] = hw["TVT"].values

    return feat


FEATURE_COLS = [
    "md",
    "md_norm",
    "gr",
    "gr_grad",
    "gr_grad2",
    "gr_pct_rank",
    "gr_mean_5",
    "gr_std_5",
    "gr_min_5",
    "gr_max_5",
    "gr_range_5",
    "gr_mean_15",
    "gr_std_15",
    "gr_min_15",
    "gr_max_15",
    "gr_range_15",
    "gr_mean_30",
    "gr_std_30",
    "gr_min_30",
    "gr_max_30",
    "gr_range_30",
    "gr_mean_60",
    "gr_std_60",
    "gr_min_60",
    "gr_max_60",
    "gr_range_60",
    "gr_mean_120",
    "gr_std_120",
    "gr_min_120",
    "gr_max_120",
    "gr_range_120",
    "xcorr_best_lag",
    "xcorr_best_corr",
    "xcorr_mean_corr",
    "dtw_dist",
    "dtw_norm_dist",
    "tw_gr",
    "gr_tw_diff",
    "gr_tw_ratio",
    "TVT_input",
    "TVT_input_missing",
    "inclination_deg",
    "azimuth_deg",
    "dls",
    "curvature",
]


def smooth_tvt(preds: np.ndarray, window: int = 31, poly: int = 3) -> np.ndarray:
    if len(preds) < window:
        return preds
    from scipy.signal import savgol_filter

    return savgol_filter(preds, window_length=window, polyorder=poly)
