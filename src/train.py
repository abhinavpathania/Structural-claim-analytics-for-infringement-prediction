from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .dataset import load_pairs_csv, build_feature_table
from .feature_builder import FeatureBuilder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_pairs_5k.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def regression_report(y_true, y_pred):
    """Generate comprehensive regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    # Distribution of predictions
    bins = [0, 20, 50, 75, 100]
    bin_labels = ["0-20%", "21-50%", "51-75%", "76-100%"]
    pred_dist = {}
    true_dist = {}

    for i in range(len(bins) - 1):
        mask_pred = (y_pred >= bins[i]) & (y_pred < bins[i+1])
        mask_true = (y_true >= bins[i]) & (y_true < bins[i+1])
        pred_dist[bin_labels[i]] = int(np.sum(mask_pred))
        true_dist[bin_labels[i]] = int(np.sum(mask_true))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
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

    # Convert labels to 0-100 scale if needed (handles 0-1 scale from current data)
    if y.max() <= 1.0:
        y = y * 100

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = Pipeline([
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

    # K-Fold CV for regression
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse = cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")
    cv_mae = cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")

    print(f"CV RMSE: {-cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
    print(f"CV R2: {cv_r2.mean():.4f} (+/- {cv_r2.std():.4f})")
    print(f"CV MAE: {-cv_mae.mean():.4f} (+/- {cv_mae.std():.4f})")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Clamp predictions to 0-100
    y_pred = np.clip(y_pred, 0, 100)

    report = regression_report(y_test, y_pred)

    out = {
        "cv_rmse": float(-cv_rmse.mean()),
        "cv_rmse_std": float(cv_rmse.std()),
        "cv_r2": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
        "cv_mae": float(-cv_mae.mean()),
        "cv_mae_std": float(cv_mae.std()),
        "test_metrics": report,
        "feature_columns": list(X.columns),
    }

    joblib.dump({
        "model": model,
        "feature_columns": list(X.columns),
        "is_regression": True,
        "label_scale": "0-100"
    }, MODELS / "risk_model.joblib")

    with open(REPORTS / "train_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nTest Metrics:")
    print(f"  MAE: {report['mae']:.4f}")
    print(f"  RMSE: {report['rmse']:.4f}")
    print(f"  R2: {report['r2']:.4f}")
    print("\nSaved model -> models/risk_model.joblib")
    print("Saved report -> reports/train_report.json")


if __name__ == "__main__":
    main()
