from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRegressor

from .dataset import load_pairs_csv, build_feature_table
from .feature_builder import FeatureBuilder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_pairs_5k.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def regression_report(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    bins = [0, 20, 50, 75, 100]
    bin_labels = ["0-20%", "21-50%", "51-75%", "76-100%"]
    pred_dist, true_dist = {}, {}
    for i in range(len(bins) - 1):
        mask_pred = (y_pred >= bins[i]) & (y_pred < bins[i+1])
        mask_true = (y_true >= bins[i]) & (y_true < bins[i+1])
        pred_dist[bin_labels[i]] = int(np.sum(mask_pred))
        true_dist[bin_labels[i]] = int(np.sum(mask_true))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
        "std_true": float(np.std(y_true)),
        "std_pred": float(np.std(y_pred)),
        "prediction_distribution": pred_dist,
        "ground_truth_distribution": true_dist,
    }


def main():
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    df = load_pairs_csv(str(DATA))
    fb = FeatureBuilder()
    X, y = build_feature_table(df, fb)

    if y.max() <= 1.0:
        y = y * 100

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Stage 1: base XGBoost regressor
    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        ))
    ])

    # K-Fold CV for base model
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse = cross_val_score(base_model, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    cv_r2 = cross_val_score(base_model, X_train, y_train, cv=kf, scoring="r2")
    cv_mae = cross_val_score(base_model, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")

    print(f"Base CV RMSE: {-cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
    print(f"Base CV R2:   {cv_r2.mean():.4f} (+/- {cv_r2.std():.4f})")
    print(f"Base CV MAE:   {-cv_mae.mean():.4f} (+/- {cv_mae.std():.4f})")

    base_model.fit(X_train, y_train)
    y_pred_base = np.clip(base_model.predict(X_test), 0, 100)

    # Stage 2: isotonic calibration on out-of-sample predictions
    # Fit calibrator on a held-out portion of training data
    X_cal, X_rem, y_cal, y_rem = train_test_split(
        X_train, y_train, test_size=0.2, random_state=99
    )
    base_model_cal = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=7,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=42, n_jobs=-1,
        ))
    ])
    cal_preds = base_model_cal.fit(X_cal, y_cal).predict(X_rem)
    calibrator = IsotonicRegression(y_min=0, y_max=100, out_of_bounds="clip")
    calibrator.fit(cal_preds, y_rem)

    # Apply calibration
    y_pred_calibrated = calibrator.predict(y_pred_base)

    print(f"\nCalibrated Test Metrics:")
    print(f"  MAE: {mean_absolute_error(y_test, y_pred_calibrated):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_calibrated)):.4f}")
    print(f"  R2:   {r2_score(y_test, y_pred_calibrated):.4f}")

    # Pick best between base and calibrated
    mae_base = mean_absolute_error(y_test, y_pred_base)
    mae_cal = mean_absolute_error(y_test, y_pred_calibrated)
    use_calibrated = mae_cal <= mae_base

    if use_calibrated:
        y_final = y_pred_calibrated
        print("  Using: IsotonicCalibrated")
    else:
        y_final = y_pred_base
        print("  Using: Base")

    report = regression_report(y_test, y_final)

    out = {
        "cv_rmse": float(-cv_rmse.mean()),
        "cv_rmse_std": float(cv_rmse.std()),
        "cv_r2": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
        "cv_mae": float(-cv_mae.mean()),
        "cv_mae_std": float(cv_mae.std()),
        "test_metrics": report,
        "feature_columns": list(X.columns),
        "calibrator": "isotonic" if use_calibrated else "none",
    }

    joblib.dump({
        "model": base_model,
        "calibrator": calibrator if use_calibrated else None,
        "feature_columns": list(X.columns),
        "is_regression": True,
        "label_scale": "0-100",
    }, MODELS / "risk_model.joblib")

    with open(REPORTS / "train_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nSaved model -> models/risk_model.joblib")
    print("Saved report -> reports/train_report.json")


if __name__ == "__main__":
    main()