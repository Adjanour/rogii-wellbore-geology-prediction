import sys, gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data.loader import load_all_wells
from src.features.build_features import build_features, FEATURE_COLS, smooth_tvt

DATA_DIR = Path(__file__).resolve().parents[2] / "train"
SEED = 42
N_FOLDS = 5


def build_fold_data(wells_dict, well_ids, step=1):
    X_list, y_list = [], []
    for wid in well_ids:
        hw, tw = wells_dict[wid]
        df = build_features(hw, tw, is_train=True)
        df = df.iloc[::step].reset_index(drop=True)
        avail = [c for c in FEATURE_COLS if c in df.columns]
        X_list.append(df[avail].values.astype(np.float32))
        y_list.append(df["TVT"].values.astype(np.float32))
        del df, hw, tw
        gc.collect()
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list)
    return X, y


def get_well_ids_array(wells_dict, well_ids, step=1):
    ids_list = []
    for wid in well_ids:
        hw, tw = wells_dict[wid]
        df = build_features(hw, tw, is_train=True)
        df = df.iloc[::step].reset_index(drop=True)
        ids_list.append(np.full(len(df), wid))
        del df, hw, tw
        gc.collect()
    return np.concatenate(ids_list)


def train_lgbm(params=None, step=5):
    if params is None:
        params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "n_estimators": 2000,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "random_state": SEED,
            "verbose": -1,
        }

    print("Loading wells...")
    wells = load_all_wells(DATA_DIR.parent, is_train=True)
    all_well_ids = list(wells.keys())
    print(f"Total: {len(all_well_ids)} wells (step={step})")

    md_feat_idx = FEATURE_COLS.index("md") if "md" in FEATURE_COLS else -1
    mono = [0] * len(FEATURE_COLS)
    if md_feat_idx >= 0:
        mono[md_feat_idx] = 1

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = None
    models = []
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(all_well_ids, groups=all_well_ids)):
        print(f"\n=== Fold {fold + 1}/{N_FOLDS} ===")
        train_ids = [all_well_ids[i] for i in tr_idx]
        val_ids = [all_well_ids[i] for i in va_idx]
        print(f"  Train: {len(train_ids)} wells  Val: {len(val_ids)} wells")

        print("  Building features...")
        X_train, y_train = build_fold_data(wells, train_ids, step=step)
        X_val, y_val = build_fold_data(wells, val_ids, step=step)
        print(f"  Train: {X_train.shape}  Val: {X_val.shape}")

        fold_params = {**params, "monotone_constraints": mono}
        model = lgb.LGBMRegressor(**fold_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
        )

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f"  MAE: {mae:.4f} ft  RMSE: {rmse:.4f} ft  best_iter: {model.best_iteration_}")
        fold_scores.append(
            {
                "fold": fold,
                "mae": mae,
                "rmse": rmse,
                "n_train": X_train.shape[0],
                "n_val": X_val.shape[0],
                "best_iter": model.best_iteration_,
            }
        )
        models.append(model)
        del X_train, y_train, X_val, y_val
        gc.collect()

    scores_df = pd.DataFrame(fold_scores)
    print(f"\n=== CV results ===")
    print(f"MAE:  {scores_df['mae'].mean():.4f} +/- {scores_df['mae'].std():.4f} ft")
    print(f"RMSE: {scores_df['rmse'].mean():.4f} +/- {scores_df['rmse'].std():.4f} ft")

    best_idx = int(scores_df["mae"].idxmin())
    return models[best_idx], scores_df


if __name__ == "__main__":
    model, scores = train_lgbm(step=5)
