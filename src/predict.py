"""Prediction functionality."""
from __future__ import annotations
import argparse
import joblib
import pandas as pd

from .config import CFG
from .feature_builder import FeatureBuilder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=str(CFG.paths.model_dir / CFG.train.save_name))
    ap.add_argument("--claim", required=True)
    ap.add_argument("--product", required=True)
    args = ap.parse_args()

    pack = joblib.load(args.model_path)
    model = pack["model"]
    cols = pack["feature_columns"]

    fb = FeatureBuilder(CFG.model.spacy_model, CFG.model.st_model)
    feats = fb.build(args.claim, args.product)
    X = pd.DataFrame([feats]).reindex(columns=cols).fillna(0.0)

    if hasattr(model, "predict_proba"):
        risk = float(model.predict_proba(X)[:, 1][0])
    else:
        risk = float(model.predict(X)[0])

    print(f"risk_score={risk:.4f} | risk_percent={risk*100:.2f}%")

if __name__ == "__main__":
    main()