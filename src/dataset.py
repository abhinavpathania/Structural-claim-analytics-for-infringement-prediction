from __future__ import annotations

from typing import Tuple
import pandas as pd
from .feature_builder import FeatureBuilder


def load_pairs_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"claim_text", "product_text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["label"] = df["label"].astype(float)
    return df


def build_feature_table(df: pd.DataFrame, fb: FeatureBuilder) -> Tuple[pd.DataFrame, pd.Series]:
    rows = []
    for _, r in df.iterrows():
        rows.append(fb.build_feature_vector(str(r["claim_text"]), str(r["product_text"])))
    X = pd.DataFrame(rows).fillna(0.0)
    y = df["label"]
    return X, y
