from __future__ import annotations

import os
os.environ.setdefault("HF_TOKEN", "hf_suppress")

import sys
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from .feature_builder import FeatureBuilder
from .explain import explain_single

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "risk_model.joblib"


def get_risk_category(score: float) -> str:
    """Categorize risk score into descriptive labels."""
    if score <= 20:
        return "Minimal Risk"
    elif score <= 50:
        return "Low Risk"
    elif score <= 75:
        return "Moderate Risk"
    elif score <= 90:
        return "High Risk"
    else:
        return "Critical Risk"


def main():
    ap = argparse.ArgumentParser(
        description="Predict infringement risk score for a claim-product pair"
    )
    ap.add_argument("--claim", required=True, help="Patent claim text")
    ap.add_argument("--product", required=True, help="Product description text")
    ap.add_argument("--detailed", action="store_true", help="Show detailed breakdown")
    ap.add_argument("--explain", action="store_true", help="Show SHAP feature explanations")
    args = ap.parse_args()

    pack = joblib.load(MODEL_PATH)
    model = pack["model"]
    calibrator = pack.get("calibrator")
    cols = pack["feature_columns"]

    fb = FeatureBuilder()
    feats = fb.build_feature_vector(args.claim, args.product)

    X = pd.DataFrame([feats]).reindex(columns=cols).fillna(0.0)
    score = float(model.predict(X)[0])
    score = np.clip(score, 0, 100)
    if calibrator is not None:
        score = float(calibrator.predict([score])[0])

    category = get_risk_category(score)
    semantic_sim = feats["semantic_similarity"]

    print("=" * 50)
    print("INFRINGEMENT RISK ASSESSMENT")
    print("=" * 50)
    print(f"\nRisk Score: {score:.2f}%")
    print(f"Category: {category}")
    print(f"Semantic Similarity: {semantic_sim * 100:.2f}%")

    if args.explain:
        print("\n--- SHAP Explanation ---")
        from .explain import print_explanation
        result = explain_single(args.claim, args.product, topk=8)
        print_explanation(result)

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()