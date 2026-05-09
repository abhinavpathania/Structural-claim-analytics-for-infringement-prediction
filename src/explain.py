from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
import shap

from .feature_builder import FeatureBuilder

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "risk_model.joblib"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", required=True)
    ap.add_argument("--product", required=True)
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    pack = joblib.load(MODEL_PATH)
    model = pack["model"]
    cols = pack["feature_columns"]

    fb = FeatureBuilder()
    feats = fb.build_feature_vector(args.claim, args.product)
    X = pd.DataFrame([feats]).reindex(columns=cols).fillna(0.0)

    # pipeline: scaler + xgb
    clf = model.named_steps["clf"]

    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(model.named_steps["scaler"].transform(X))

    vals = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
    contrib = sorted(zip(cols, vals[0]), key=lambda x: abs(x[1]), reverse=True)[:args.topk]

    print("Top feature contributions (by |SHAP|):")
    for k, v in contrib:
        print(f"{k:28s} {float(v):+.5f}")

if __name__ == "__main__":
    main()
