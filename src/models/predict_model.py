import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gc
import numpy as np
import pandas as pd
from src.data.loader import load_all_wells, load_horizontal, load_typewell
from src.features.build_features import build_features, FEATURE_COLS

TEST_DIR = Path(__file__).resolve().parents[2] / "test"
TRAIN_DIR = Path(__file__).resolve().parents[2] / "train"
OUTPUT = Path(__file__).resolve().parents[2] / "submission.csv"


def train_on_all(step=5):
    import xgboost as xgb

    print("Loading all training wells...")
    wells = load_all_wells(TRAIN_DIR.parent, is_train=True)
    all_ids = list(wells.keys())

    X_list, y_list = [], []
    for wid in all_ids:
        hw, tw = wells[wid]
        df = build_features(hw, tw, is_train=True)
        df = df.iloc[::step].reset_index(drop=True)
        avail = [c for c in FEATURE_COLS if c in df.columns]
        X_list.append(df[avail].values.astype(np.float32))
        y_list.append(df["TVT"].values.astype(np.float32))
        del df, hw, tw
        gc.collect()

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list)
    print(f"Training on {X.shape[0]} rows, {X.shape[1]} features")

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    model.get_booster().feature_names = list(avail)
    return model


def predict_test(model):
    print("\nPredicting test wells...")
    well_ids = sorted(set(p.stem.split("__")[0] for p in TEST_DIR.glob("*__horizontal_well.csv")))
    all_ids, all_tvts = [], []

    for wid in well_ids:
        hw = load_horizontal(TEST_DIR, wid, is_train=False)
        tw = load_typewell(TEST_DIR, wid, is_train=False)

        # Track which rows had NaN TVT_input (those are the ones we need to predict)
        nan_mask = hw["TVT_input"].isna().values

        df = build_features(hw, tw, is_train=False)
        avail = [c for c in FEATURE_COLS if c in df.columns]
        X = df[avail].values.astype(np.float32)

        preds = model.predict(X)

        # Only keep predictions for rows that originally had NaN TVT_input
        keep_idx = np.where(nan_mask)[0]
        for idx in keep_idx:
            all_ids.append(f"{wid}_{idx}")
            all_tvts.append(preds[idx])

        print(f"  {wid}: {len(keep_idx)}/{len(hw)} rows predicted")
        del hw, tw, df, X, preds
        gc.collect()

    sub = pd.DataFrame({"id": all_ids, "tvt": all_tvts})
    sub.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(sub)} predictions to {OUTPUT}")
    print(sub.head())
    return sub


if __name__ == "__main__":
    model = train_on_all(step=1)
    sub = predict_test(model)
