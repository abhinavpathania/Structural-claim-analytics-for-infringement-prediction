from __future__ import annotations
from .feature_builder import FeatureBuilder

def main():
    fb = FeatureBuilder()
    claim = "A system configured to transmit data to a remote server and process received signals."
    product = "The device processes input signals and sends data to a cloud server."
    feats = fb.build_feature_vector(claim, product)
    for k in sorted(feats.keys()):
        print(f"{k:28s} = {feats[k]}")
    print("\nSimilarity %:", round(feats["semantic_similarity"] * 100, 2))

if __name__ == "__main__":
    main()
