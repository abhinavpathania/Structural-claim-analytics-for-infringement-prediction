"""Model explanation and interpretability."""
from __future__ import annotations
import argparse
import joblib
import pandas as pd
import shap

from .config import CFG
from .feature_builder import FeatureBuilder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=str(CFG.paths.model_dir / CFG.train.save_name))
    ap.add_argument("--claim", required=True)
    ap.add_argument("--product", required=True)
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    pack = joblib.load(args.model_path)
    model = pack["model"]
    cols = pack["feature_columns"]

    fb = FeatureBuilder(CFG.model.spacy_model, CFG.model.st_model)
    feats = fb.build(args.claim, args.product)
    X = pd.DataFrame([feats]).reindex(columns=cols).fillna(0.0)

    # Works best for tree models; fallback to KernelExplainer if needed
    try:
        explainer = shap.Explainer(model)
        sv = explainer(X)
        vals = sv.values[0]
    except Exception:
        explainer = shap.KernelExplainer(lambda z: model.predict_proba(pd.DataFrame(z, columns=cols))[:,1], X)
        sv = explainer.shap_values(X)
        vals = sv[0] if isinstance(sv, list) else sv

    contrib = sorted(zip(cols, vals), key=lambda x: abs(x[1]), reverse=True)[:args.topk]
    print("Top feature contributions (by |SHAP|):")
    for k, v in contrib:
        print(f"{k:28s}  {float(v):+.5f}")

if __name__ == "__main__":
    main()