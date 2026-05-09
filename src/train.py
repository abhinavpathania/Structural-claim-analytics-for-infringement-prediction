from __future__ import annotations

import json
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .dataset import load_pairs_csv, build_feature_table
from .feature_builder import FeatureBuilder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_pairs_5k.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

def main():
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    df = load_pairs_csv(str(DATA))
    fb = FeatureBuilder()
    X, y = build_feature_table(df, fb)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        ))
    ])

    # CV
    n_splits = min(3, y_train.value_counts().min())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    print("CV F1:", cv_f1, "Mean:", cv_f1.mean())

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = None
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    out = {
        "cv_f1_scores": cv_f1.tolist(),
        "test_report": report,
        "test_auc": float(auc),
        "feature_columns": list(X.columns),
    }

    joblib.dump({"model": model, "feature_columns": list(X.columns)}, MODELS / "risk_model.joblib")
    with open(REPORTS / "train_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved model -> models/risk_model.joblib")
    print("Saved report -> reports/train_report.json")
    print("Test AUC:", auc)

if __name__ == "__main__":
    main()
