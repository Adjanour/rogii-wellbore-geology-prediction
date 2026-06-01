import sys, gc, pickle, dataclasses
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from src.data.loader import load_all_wells, load_horizontal, load_typewell
from src.features.build_features import build_features, FEATURE_COLS, smooth_tvt

TEST_DIR = Path(__file__).resolve().parents[2] / "test"
TRAIN_DIR = Path(__file__).resolve().parents[2] / "train"
OUTPUT = Path(__file__).resolve().parents[2] / "submission.csv"


@dataclasses.dataclass
class TVTInferenceModel:
    model: lgb.LGBMRegressor
    feature_cols: list
    sg_window: int = 31
    sg_poly: int = 3
    oof_rmse: float = 0.0
    n_train_wells: int = 0

    def predict(self, well_df: pd.DataFrame, typewell_df, smooth: bool = True) -> np.ndarray:
        feat = build_features(well_df, typewell_df, is_train=False)
        X_inf = feat[self.feature_cols].fillna(feat[self.feature_cols].median())
        preds = self.model.predict(X_inf)
        if smooth:
            preds = smooth_tvt(preds, self.sg_window, self.sg_poly)
        return preds


def train_on_all(step=5):
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

    md_idx = FEATURE_COLS.index("md") if "md" in FEATURE_COLS else -1
    mono = [0] * len(FEATURE_COLS)
    if md_idx >= 0:
        mono[md_idx] = 1

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        metric="rmse",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        monotone_constraints=mono,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)
    print("Final model trained on all data.")
    return model, avail


def predict_test(model, avail):
    print("\nPredicting test wells...")
    well_ids = sorted(set(p.stem.split("__")[0] for p in TEST_DIR.glob("*__horizontal_well.csv")))
    all_ids, all_tvts = [], []

    for wid in well_ids:
        hw = load_horizontal(TEST_DIR, wid, is_train=False)
        tw = load_typewell(TEST_DIR, wid, is_train=False)
        nan_mask = hw["TVT_input"].isna().values
        df = build_features(hw, tw, is_train=False)
        X = df[avail].values.astype(np.float32)
        preds = smooth_tvt(model.predict(X))

        keep_idx = np.where(nan_mask)[0]
        for idx in keep_idx:
            all_ids.append(f"{wid}_{idx}")
            all_tvts.append(preds[idx])
        print(f"  {wid}: {len(keep_idx)}/{len(hw)} rows")
        del hw, tw, df, X, preds
        gc.collect()

    sub = pd.DataFrame({"id": all_ids, "tvt": all_tvts})
    sub.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(sub)} predictions to {OUTPUT}")
    print(sub.head())
    return sub


def export_onnx(model, avail, output_dir):
    try:
        from skl2onnx import convert_sklearn, update_registered_converter
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm

        update_registered_converter(
            lgb.LGBMRegressor,
            "LightGbmLGBMRegressor",
            calculate_linear_regressor_output_shapes,
            convert_lightgbm,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )
        initial_type = [("float_input", FloatTensorType([None, len(avail)]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=17)
        onnx_path = output_dir / "tvt_model.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX saved → {onnx_path}  ({onnx_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"ONNX export failed ({e}) — pickle remains available")


if __name__ == "__main__":
    model, avail = train_on_all(step=1)
    sub = predict_test(model, avail)

    export_dir = Path(__file__).resolve().parents[2]
    inference_model = TVTInferenceModel(
        model=model,
        feature_cols=avail,
        sg_window=31,
        sg_poly=3,
        oof_rmse=0.0,
        n_train_wells=len(list((TRAIN_DIR.parent / "train").glob("*__horizontal_well.csv"))),
    )
    pkl_path = export_dir / "tvt_model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(inference_model, f, protocol=5)
    print(f"Pickle saved → {pkl_path}  ({pkl_path.stat().st_size / 1024:.1f} KB)")

    export_onnx(model, avail, export_dir)
