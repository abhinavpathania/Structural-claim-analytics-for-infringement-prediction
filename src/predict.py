from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from .feature_builder import FeatureBuilder

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "risk_model.joblib"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", required=True)
    ap.add_argument("--product", required=True)
    args = ap.parse_args()

    pack = joblib.load(MODEL_PATH)
    model = pack["model"]
    cols = pack["feature_columns"]

    fb = FeatureBuilder()
    feats = fb.build_feature_vector(args.claim, args.product)

    X = pd.DataFrame([feats]).reindex(columns=cols).fillna(0.0)
    risk = float(model.predict_proba(X)[:, 1][0])

    print(f"Risk score: {risk:.4f} ({risk*100:.2f}%)")
    print("Similarity %:", round(feats["semantic_similarity"] * 100, 2))

if __name__ == "__main__":
    main()