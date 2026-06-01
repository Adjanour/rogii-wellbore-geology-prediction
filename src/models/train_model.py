import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data.loader import load_all_wells
from src.features.build_features import build_features, FEATURE_COLS

DATA_DIR = Path(__file__).resolve().parents[2] / "train"


def build_fold_data(wells_dict, well_ids, step=1, is_train=True):
    X_list, y_list = [], []
    for wid in well_ids:
        hw, tw = wells_dict[wid]
        df = build_features(hw, tw, is_train=is_train)
        df = df.iloc[::step].reset_index(drop=True)
        avail = [c for c in FEATURE_COLS if c in df.columns]
        X_list.append(df[avail].values.astype(np.float32))
        if is_train:
            y_list.append(df["TVT"].values.astype(np.float32))
        del df, hw, tw
        gc.collect()
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list) if y_list else None
    return X, y


def train_xgboost(params=None, step=5):
    import xgboost as xgb

    if params is None:
        params = {
            "n_estimators": 500,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "random_state": 42,
        }

    print("Loading wells...")
    wells = load_all_wells(DATA_DIR.parent, is_train=True)
    all_well_ids = list(wells.keys())
    print(f"Total: {len(all_well_ids)} wells (step={step})")

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_well_ids)):
        print(f"\n=== Fold {fold + 1} ===")
        train_ids = [all_well_ids[i] for i in train_idx]
        val_ids = [all_well_ids[i] for i in val_idx]

        print("  Building train features...")
        X_train, y_train = build_fold_data(wells, train_ids, step=step)
        print(f"  Train: {X_train.shape}")

        print("  Building val features...")
        X_val, y_val = build_fold_data(wells, val_ids, step=step)
        print(f"  Val:   {X_val.shape}")

        model = xgb.XGBRegressor(**params, verbosity=0)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        naive_mae = mean_absolute_error(y_val, X_val[:, FEATURE_COLS.index("TVT_input")])

        print(f"  MAE: {mae:.4f} ft  (naive: {naive_mae:.4f})")
        print(f"  RMSE: {rmse:.4f} ft")

        fold_scores.append(
            {
                "fold": fold,
                "mae": mae,
                "rmse": rmse,
                "naive_mae": naive_mae,
                "n_train": X_train.shape[0],
                "n_val": X_val.shape[0],
            }
        )
        models.append(model)
        del X_train, y_train, X_val, y_val
        gc.collect()

    scores_df = pd.DataFrame(fold_scores)
    print(f"\n=== CV results ===")
    print(f"MAE:  {scores_df['mae'].mean():.4f} +/- {scores_df['mae'].std():.4f} ft")
    print(f"RMSE: {scores_df['rmse'].mean():.4f} +/- {scores_df['rmse'].std():.4f} ft")
    print(f"Naive: {scores_df['naive_mae'].mean():.4f} ft")

    best_idx = int(scores_df["mae"].idxmin())
    return models[best_idx], scores_df


if __name__ == "__main__":
    model, scores = train_xgboost(step=5)
