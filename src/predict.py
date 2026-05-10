from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from .feature_builder import FeatureBuilder

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
    args = ap.parse_args()

    pack = joblib.load(MODEL_PATH)
    model = pack["model"]
    cols = pack["feature_columns"]

    fb = FeatureBuilder()
    feats = fb.build_feature_vector(args.claim, args.product)

    X = pd.DataFrame([feats]).reindex(columns=cols).fillna(0.0)
    score = float(model.predict(X)[0])
    score = np.clip(score, 0, 100)

    category = get_risk_category(score)
    semantic_sim = feats["semantic_similarity"]

    print("=" * 50)
    print("INFRINGEMENT RISK ASSESSMENT")
    print("=" * 50)
    print(f"\nRisk Score: {score:.2f}%")
    print(f"Category: {category}")
    print(f"Semantic Similarity: {semantic_sim * 100:.2f}%")

    if args.detailed:
        print("\n--- Feature Analysis ---")
        print(f"  Claim word count: {feats['claim_word_count']:.0f}")
        print(f"  Claim abstraction: {feats['claim_abstraction_score']:.0f}")
        print(f"  Claim dependency depth: {feats['claim_dep_depth_mean']:.2f}")
        print(f"  Product components: {feats['prod_component_terms']:.0f}")
        print(f"  Functional density: {feats['prod_functional_density']:.3f}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()