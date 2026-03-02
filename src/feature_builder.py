"""Feature building and engineering."""
from __future__ import annotations
from typing import Dict
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .text_features import build_claim_stats, build_product_stats, to_feature_dict

class FeatureBuilder:
    def __init__(self, spacy_model: str, st_model: str):
        self.nlp = spacy.load(spacy_model, disable=["ner"])  # faster + consistent
        self.embedder = SentenceTransformer(st_model)

    def semantic_similarity(self, a: str, b: str) -> float:
        ea = self.embedder.encode([a], normalize_embeddings=True)
        eb = self.embedder.encode([b], normalize_embeddings=True)
        return float(cosine_similarity(ea, eb)[0][0])

    def build(self, claim_text: str, product_text: str) -> Dict[str, float]:
        cs = build_claim_stats(self.nlp, claim_text)
        ps = build_product_stats(self.nlp, product_text)
        feats = to_feature_dict(cs, ps)
        feats["semantic_similarity"] = self.semantic_similarity(claim_text, product_text)
        return feats