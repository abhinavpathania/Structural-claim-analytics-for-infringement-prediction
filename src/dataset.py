"""Dataset loading and preprocessing."""
from __future__ import annotations
from typing import Tuple
import pandas as pd

def load_pairs_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"claim_text","product_text","label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df

def build_feature_table(df: pd.DataFrame, fb) -> Tuple[pd.DataFrame, pd.Series]:
    rows = []
    for _, r in df.iterrows():
        feats = fb.build(str(r["claim_text"]), str(r["product_text"]))
        rows.append(feats)
    X = pd.DataFrame(rows).fillna(0.0)
    y = df["label"].astype(int)
    return X, y