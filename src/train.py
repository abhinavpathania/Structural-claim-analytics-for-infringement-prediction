"""Model training functionality."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .config import CFG
from .utils import get_logger, ensure_dir
from .feature_builder import FeatureBuilder
from .dataset import load_pairs_csv, build_feature_table

log = get_logger()

def make_model(name: str):
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=CFG.model.random_state))
        ])
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=CFG.model.random_state,
            n_jobs=-1,
        )
    if name == "xgb":
        return XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=CFG.model.random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )
    raise ValueError(f"Unknown model: {name}")

def main():
    paths = CFG.paths
    ensure_dir(paths.model_dir)
    ensure_dir(paths.reports_dir)

    df = load_pairs_csv(str(paths.data_dir / "sample_pairs.csv"))
    fb = FeatureBuilder(CFG.model.spacy_model, CFG.model.st_model)
    X, y = build_feature_table(df, fb)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CFG.train.test_size, random_state=CFG.model.random_state, stratify=y
    )

    model = make_model(CFG.train.model_name)

    cv = StratifiedKFold(n_splits=min(CFG.train.cv_folds, y_train.value_counts().min()), shuffle=True, random_state=CFG.model.random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    log.info(f"CV F1 mean={cv_scores.mean():.3f} std={cv_scores.std():.3f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        log.info(f"Test ROC-AUC={auc:.3f}")

    out_model = paths.model_dir / CFG.train.save_name
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, out_model)
    log.info(f"Saved model to {out_model}")

    out_report = paths.reports_dir / "train_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump({"cv_f1_scores": cv_scores.tolist(), "test_report": report, "test_roc_auc": auc}, f, indent=2)
    log.info(f"Saved report to {out_report}")

if __name__ == "__main__":
    main()