from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import re
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

FUNCTION_VERBS = {
    "transmit", "process", "detect", "generate", "receive", "store", "execute", "send",
    "synchronize", "sync", "update", "encrypt", "decrypt", "authenticate", "compute",
    "encode", "decode", "route", "schedule", "allocate"
}

STRUCTURAL_NOUNS = {
    "processor", "memory", "module", "unit", "server", "client", "interface", "sensor",
    "controller", "database", "network", "bus", "circuit", "engine", "application", "device"
}

ABSTRACTION_PATTERNS = [
    r"\bconfigured to\b",
    r"\badapted to\b",
    r"\boperable to\b",
    r"\barranged to\b",
    r"\bfor performing\b"
]

CLAUSE_MARKERS = ["wherein", "whereby", "such that", "in response to", "based on", "according to"]

INTERACTION_PATTERNS = [
    r"\bcommunicat(?:e|ing|es)\b",
    r"\btransmit(?:s|ting)?\b",
    r"\bsend(?:s|ing)?\b",
    r"\breceiv(?:e|es|ing)\b",
    r"\bsynchroniz(?:e|es|ing)\b",
    r"\bupdate(?:s|ing)?\b",
    r"\bconnect(?:s|ing)?\b"
]

def _count_patterns(patterns: List[str], text: str) -> int:
    s = text.lower()
    return sum(1 for p in patterns if re.search(p, s))

def _count_markers(markers: List[str], text: str) -> int:
    s = text.lower()
    return sum(s.count(m) for m in markers)

def _dependency_depths(doc) -> List[int]:
    depths = []
    for t in doc:
        if t.is_punct or t.is_space:
            continue
        depths.append(len(list(t.ancestors)))
    return depths if depths else [0]

@dataclass(frozen=True)
class ClaimFeatures:
    word_count: int
    token_count: int
    clause_markers: int
    dep_depth_max: int
    dep_depth_mean: float
    func_verb_density: float
    struct_noun_density: float
    abstraction_score: int
    means_plus_function: int
    interaction_density: float

@dataclass(frozen=True)
class ProductFeatures:
    token_count: int
    component_terms: int
    interaction_terms: int
    functional_terms: int
    functional_density: float

class FeatureBuilder:
    def __init__(self, spacy_model: str = "en_core_web_sm", st_model: str = "all-MiniLM-L6-v2"):
        self.nlp = spacy.load(spacy_model, disable=["ner"])
        self.embedder = SentenceTransformer(st_model)

    def build_claim_features(self, claim_text: str) -> ClaimFeatures:
        doc = self.nlp(claim_text)
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        words = [t for t in tokens if t.is_alpha or t.like_num]

        depths = _dependency_depths(doc)

        func_verbs = [t for t in tokens if t.pos_ == "VERB" and t.lemma_.lower() in FUNCTION_VERBS]
        struct_nouns = [t for t in tokens if t.pos_ in {"NOUN", "PROPN"} and t.lemma_.lower() in STRUCTURAL_NOUNS]

        token_count = len(tokens)
        func_density = (len(func_verbs) / token_count) if token_count else 0.0
        struct_density = (len(struct_nouns) / token_count) if token_count else 0.0

        abstraction_score = _count_patterns(ABSTRACTION_PATTERNS, claim_text)
        clause_markers = _count_markers(CLAUSE_MARKERS, claim_text)
        means_plus_function = 1 if re.search(r"\bmeans for\b", claim_text.lower()) else 0

        interaction_hits = sum(1 for p in INTERACTION_PATTERNS if re.search(p, claim_text.lower()))
        interaction_density = (interaction_hits / max(1, token_count)) * 10.0

        return ClaimFeatures(
            word_count=len(words),
            token_count=token_count,
            clause_markers=clause_markers,
            dep_depth_max=int(np.max(depths)),
            dep_depth_mean=float(np.mean(depths)),
            func_verb_density=float(func_density),
            struct_noun_density=float(struct_density),
            abstraction_score=int(abstraction_score),
            means_plus_function=int(means_plus_function),
            interaction_density=float(interaction_density),
        )

    def build_product_features(self, product_text: str) -> ProductFeatures:
        doc = self.nlp(product_text)
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        token_count = len(tokens)

        component_terms = sum(1 for t in tokens if t.pos_ in {"NOUN", "PROPN"} and t.lemma_.lower() in STRUCTURAL_NOUNS)
        interaction_terms = sum(
            1 for t in tokens
            if t.pos_ == "VERB" and re.search(r"(send|receive|transmit|sync|synchroniz|connect|communicat|update)", t.lemma_.lower())
        )
        functional_terms = sum(1 for t in tokens if t.pos_ == "VERB" and t.lemma_.lower() in FUNCTION_VERBS)
        functional_density = (functional_terms / token_count) if token_count else 0.0

        return ProductFeatures(
            token_count=token_count,
            component_terms=int(component_terms),
            interaction_terms=int(interaction_terms),
            functional_terms=int(functional_terms),
            functional_density=float(functional_density),
        )

    def semantic_similarity(self, claim_text: str, product_text: str) -> float:
        a = self.embedder.encode([claim_text], normalize_embeddings=True)
        b = self.embedder.encode([product_text], normalize_embeddings=True)
        return float(cosine_similarity(a, b)[0][0])

    def build_feature_vector(self, claim_text: str, product_text: str) -> Dict[str, float]:
        c = self.build_claim_features(claim_text)
        p = self.build_product_features(product_text)
        s = self.semantic_similarity(claim_text, product_text)

        return {
            "claim_word_count": float(c.word_count),
            "claim_token_count": float(c.token_count),
            "claim_clause_markers": float(c.clause_markers),
            "claim_dep_depth_max": float(c.dep_depth_max),
            "claim_dep_depth_mean": float(c.dep_depth_mean),
            "claim_func_verb_density": float(c.func_verb_density),
            "claim_struct_noun_density": float(c.struct_noun_density),
            "claim_abstraction_score": float(c.abstraction_score),
            "claim_means_plus_function": float(c.means_plus_function),
            "claim_interaction_density": float(c.interaction_density),
            "prod_token_count": float(p.token_count),
            "prod_component_terms": float(p.component_terms),
            "prod_interaction_terms": float(p.interaction_terms),
            "prod_functional_terms": float(p.functional_terms),
            "prod_functional_density": float(p.functional_density),
            "semantic_similarity": float(s),
        }
