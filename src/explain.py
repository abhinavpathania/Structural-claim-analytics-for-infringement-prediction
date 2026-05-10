from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import shap

from .feature_builder import FeatureBuilder

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "risk_model.joblib"


FEATURE_GROUPS = {
    "Claim -- structure": [
        "claim_word_count", "claim_token_count", "claim_dep_depth_max", "claim_dep_depth_mean",
    ],
    "Claim -- language": [
        "claim_clause_markers", "claim_abstraction_score", "claim_means_plus_function",
        "claim_func_verb_density", "claim_struct_noun_density", "claim_interaction_density",
    ],
    "Product -- structure": [
        "prod_token_count", "prod_component_terms", "prod_functional_terms",
    ],
    "Product -- language": [
        "prod_interaction_terms", "prod_functional_density",
    ],
    "Cross-pair similarity": [
        "semantic_similarity",
    ],
}


def _group_shap(vals, cols):
    grouped = {k: 0.0 for k in FEATURE_GROUPS}
    for col, val in zip(cols, vals):
        for group, members in FEATURE_GROUPS.items():
            if col in members:
                grouped[group] += abs(val)
                break
    return grouped


def explain_single(claim: str, product: str, topk: int = 8, base_value: float = None):
    pack = joblib.load(MODEL_PATH)
    model = pack["model"]
    calibrator = pack.get("calibrator")
    cols = pack["feature_columns"]

    fb = FeatureBuilder()
    feats = fb.build_feature_vector(claim, product)
    X = pd.DataFrame([feats]).reindex(columns=cols).fillna(0.0)

    scaler = model.named_steps["scaler"]
    regressor = model.named_steps["reg"]
    X_scaled = scaler.transform(X)

    if base_value is not None:
        base_score = float(base_value)
    else:
        b = regressor.base_score
        base_score = float(b) if b is not None else 50.0

    explainer = shap.TreeExplainer(regressor, feature_names=cols)
    shap_vals = explainer.shap_values(X_scaled)[0]

    individual = sorted(zip(cols, shap_vals), key=lambda x: abs(x[1]), reverse=True)
    grouped = _group_shap(shap_vals, cols)

    raw_score = float(model.predict(X)[0])
    raw_score = np.clip(raw_score, 0, 100)
    calibrated_score = float(calibrator.predict([raw_score])[0]) if calibrator else raw_score

    return {
        "base_value": base_score,
        "prediction": calibrated_score,
        "shap_values": dict(individual[:topk]),
        "grouped_shap": grouped,
    }


def print_explanation(result: dict, indent: str = "  "):
    print(f"{indent}Base value: {result['base_value']:.2f}")
    print(f"{indent}Prediction:  {result['prediction']:.2f}")
    print(f"{indent}Net SHAP:    {sum(result['shap_values'].values()):+.2f}")
    print(f"\n{indent}Top features (individual):")
    for k, v in result["shap_values"].items():
        direction = "+" if v > 0 else "-"
        print(f"{indent}  {direction} {k:<38} {v:+.5f}")

    print(f"\n{indent}By group:")
    for group, val in sorted(result["grouped_shap"].items(), key=lambda x: -x[1]):
        bar = "#" * min(int(val / 2), 40)
        print(f"{indent}  {group:<28} {val:6.3f} {bar}")


def main():
    ap = argparse.ArgumentParser(description="SHAP feature explanations for infringement risk")
    ap.add_argument("--claim", required=True, help="Patent claim text")
    ap.add_argument("--product", required=True, help="Product description text")
    ap.add_argument("--topk", type=int, default=8, help="Number of top features to show")
    ap.add_argument("--groups", action="store_true", help="Show grouped contribution summary")
    args = ap.parse_args()

    result = explain_single(args.claim, args.product, topk=args.topk)

    print("=" * 60)
    print("SHAP EXPLANATION")
    print("=" * 60)
    print_explanation(result)

    if args.groups:
        print("\nGrouped contribution (|SHAP|):")
        total = sum(result["grouped_shap"].values())
        for group, val in sorted(result["grouped_shap"].items(), key=lambda x: -x[1]):
            pct = (val / total * 100) if total > 0 else 0
            print(f"  {group:<28} {val:6.3f} ({pct:5.1f}%)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()