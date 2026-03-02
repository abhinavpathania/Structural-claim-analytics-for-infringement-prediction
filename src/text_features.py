"""Text feature extraction and processing."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import re
import numpy as np
import spacy

FUNCTION_VERBS = {
    "transmit","process","detect","generate","receive","store","execute","send",
    "synchronize","update","encrypt","decrypt","authenticate","compute","render",
    "encode","decode","route","schedule","allocate","classify","predict"
}

STRUCTURAL_NOUNS = {
    "processor","memory","module","unit","server","client","interface","sensor",
    "controller","database","network","bus","circuit","engine","application","device"
}

ABSTRACTION_PATTERNS = [
    r"\bconfigured to\b",
    r"\badapted to\b",
    r"\bfor performing\b",
    r"\boperable to\b",
    r"\barranged to\b"
]

DEPENDENT_MARKERS = [
    "wherein", "whereby", "such that", "in response to", "based on", "according to"
]

@dataclass(frozen=True)
class ClaimStats:
    word_count: int
    token_count: int
    clause_markers: int
    dependency_depth_max: int
    dependency_depth_mean: float
    functional_verb_density: float
    structural_noun_density: float
    abstraction_score: int
    has_means_plus_function: int

def _dependency_depths(doc) -> List[int]:
    depths = []
    for t in doc:
        if t.is_punct:
            continue
        depths.append(len(list(t.ancestors)))
    return depths or [0]

def _count_regex(patterns: List[str], text: str) -> int:
    s = text.lower()
    return sum(1 for p in patterns if re.search(p, s))

def _count_markers(text: str) -> int:
    s = text.lower()
    return sum(s.count(m) for m in DEPENDENT_MARKERS)

def build_claim_stats(nlp: spacy.language.Language, claim_text: str) -> ClaimStats:
    doc = nlp(claim_text)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    words = [t for t in tokens if t.is_alpha or t.like_num]

    depths = _dependency_depths(doc)
    verbs = [t for t in tokens if t.pos_ == "VERB" and t.lemma_.lower() in FUNCTION_VERBS]
    nouns = [t for t in tokens if t.pos_ in {"NOUN","PROPN"} and t.lemma_.lower() in STRUCTURAL_NOUNS]

    abstraction_score = _count_regex(ABSTRACTION_PATTERNS, claim_text)
    clause_markers = _count_markers(claim_text)
    means_plus_function = 1 if re.search(r"\bmeans for\b", claim_text.lower()) else 0

    token_count = len(tokens)
    functional_verb_density = (len(verbs) / token_count) if token_count else 0.0
    structural_noun_density = (len(nouns) / token_count) if token_count else 0.0

    return ClaimStats(
        word_count=len(words),
        token_count=token_count,
        clause_markers=clause_markers,
        dependency_depth_max=int(np.max(depths)),
        dependency_depth_mean=float(np.mean(depths)),
        functional_verb_density=float(functional_verb_density),
        structural_noun_density=float(structural_noun_density),
        abstraction_score=int(abstraction_score),
        has_means_plus_function=int(means_plus_function),
    )

@dataclass(frozen=True)
class ProductStats:
    token_count: int
    component_terms: int
    interaction_terms: int
    functional_terms: int
    functional_density: float

INTERACTION_TERMS = {"send","receive","transmit","sync","synchronize","connect","communicate","stream","route"}
COMPONENT_HINTS = STRUCTURAL_NOUNS

def build_product_stats(nlp: spacy.language.Language, product_text: str) -> ProductStats:
    doc = nlp(product_text)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    token_count = len(tokens)

    comp = sum(1 for t in tokens if t.pos_ in {"NOUN","PROPN"} and t.lemma_.lower() in COMPONENT_HINTS)
    inter = sum(1 for t in tokens if t.pos_ == "VERB" and t.lemma_.lower() in INTERACTION_TERMS)
    func = sum(1 for t in tokens if t.pos_ == "VERB" and t.lemma_.lower() in FUNCTION_VERBS)

    return ProductStats(
        token_count=token_count,
        component_terms=int(comp),
        interaction_terms=int(inter),
        functional_terms=int(func),
        functional_density=float((func / token_count) if token_count else 0.0),
    )

def to_feature_dict(claim: ClaimStats, product: ProductStats) -> Dict[str, float]:
    return {
        "claim_word_count": float(claim.word_count),
        "claim_token_count": float(claim.token_count),
        "claim_clause_markers": float(claim.clause_markers),
        "claim_dep_depth_max": float(claim.dependency_depth_max),
        "claim_dep_depth_mean": float(claim.dependency_depth_mean),
        "claim_func_verb_density": float(claim.functional_verb_density),
        "claim_struct_noun_density": float(claim.structural_noun_density),
        "claim_abstraction_score": float(claim.abstraction_score),
        "claim_means_plus_function": float(claim.has_means_plus_function),
        "prod_token_count": float(product.token_count),
        "prod_component_terms": float(product.component_terms),
        "prod_interaction_terms": float(product.interaction_terms),
        "prod_functional_terms": float(product.functional_terms),
        "prod_functional_density": float(product.functional_density),
    }