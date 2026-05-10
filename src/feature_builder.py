from __future__ import annotations

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_TOKEN"] = "hf_dummy_token_suppress_warning"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import re
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Domain vocabulary from dataset_generation.py ────────────────────────────
DOMAINS = {
    "data_sync": {
        "components": [
            "server", "client", "database", "processor", "memory", "network interface",
            "controller", "network", "bus", "engine", "module"
        ],
        "actions": [
            "synchronize", "sync", "transmit", "receive", "update", "process", "store",
            "send", "allocate"
        ],
        "objects": ["record", "message", "event", "packet", "state", "update", "data"]
    },
    "motion_alert": {
        "components": [
            "sensor", "camera", "processor", "controller", "wireless module", "memory",
            "circuit", "module", "device", "chip", "antenna", "receiver", "transmitter"
        ],
        "actions": [
            "detect", "generate", "transmit", "receive", "process", "store", "monitor",
            "regulate", "convert", "control"
        ],
        "objects": ["motion", "frame", "alert", "signal", "event", "image", "video"]
    },
    "auth_security": {
        "components": [
            "processor", "memory", "secure module", "server", "client device", "interface",
            "controller", "module", "device", "chip", "circuit", "engine"
        ],
        "actions": [
            "authenticate", "encrypt", "decrypt", "store", "verify", "transmit", "send",
            "receive", "process", "encode", "decode"
        ],
        "objects": ["key", "token", "credential", "request", "response", "data", "packet"]
    },
    "power_hw": {
        "components": [
            "circuit", "voltage regulator", "power module", "sensor", "controller",
            "module", "device", "chip", "processor", "interface"
        ],
        "actions": [
            "regulate", "monitor", "convert", "control", "process", "receive", "transmit",
            "store"
        ],
        "objects": ["voltage", "current", "power state", "signal", "power", "data"]
    },
    "analytics_ui": {
        "components": [
            "dashboard", "application", "ui module", "report engine", "database",
            "server", "interface", "module", "device", "processor", "memory"
        ],
        "actions": [
            "render", "display", "generate", "aggregate", "process", "store",
            "transmit", "receive", "update"
        ],
        "objects": ["chart", "report", "metric", "dashboard", "data", "result", "alert"]
    }
}

OUT_OF_DOMAIN_VOCAB = {
    "pharmaceutical": ["compound", "formulation", "carrier", "administer", "therapeutic",
                       "ingredient", "solubility", "absorption", "dosage"],
    "construction": ["cement", "aggregate", "beam", "foundation", "structural",
                     "load-bearing", "concrete", "frame", "scaffold"],
    "ecommerce": ["cart", "checkout", "inventory", "shipment", "customer", "order", "pricing",
                  "transaction", "merchant"],
    "finance": ["portfolio", "asset", "dividend", "equity", "bond", "hedge", "derivative",
                "tranche", "amortization"],
    "marketing": ["campaign", "conversion", "engagement", "demographic", "targeting",
                  "impression", "clickthrough"],
    "food": ["flavor", "preservative", "shelf-life", "calorie", "nutrient", "texture",
             "sweetener", "additive"],
    "textiles": ["fiber", "weave", "fabric", "yarn", "dye", "blend", "thread"],
    "agriculture": ["soil", "irrigation", "harvest", "crop", "fertilizer", "pest"],
}

OUT_OF_DOMAIN_KEYWORDS = {term for vocab in OUT_OF_DOMAIN_VOCAB.values() for term in vocab}

FUNCTION_VERBS = {
    "transmit", "process", "detect", "generate", "receive", "store", "execute", "send",
    "synchronize", "sync", "update", "encrypt", "decrypt", "authenticate", "compute",
    "encode", "decode", "route", "schedule", "allocate", "verify", "convert", "regulate",
    "monitor", "control", "render", "display", "aggregate"
}

STRUCTURAL_NOUNS = {
    "server", "client", "database", "processor", "memory", "interface", "controller",
    "network", "bus", "engine", "module", "sensor", "camera", "device", "chip",
    "antenna", "regulator", "wireless", "dashboard", "application", "unit", "circuit",
    "keypad", "screen", "buffer", "transmitter", "receiver", "voltage", "current",
    "frame", "buffer", "signal"
}

ABSTRACTION_PATTERNS = [
    r"\bconfigured to\b", r"\badapted to\b", r"\boperable to\b",
    r"\barranged to\b", r"\bfor performing\b", r"\bcomprising\b", r"\bincluding\b",
    r"\boperably coupled\b", r"\bremote\b", r"\bpredetermined\b"
]

CLAUSE_MARKERS = ["wherein", "whereby", "such that", "in response to", "based on", "according to"]

INTERACTION_PATTERNS = [
    r"\bcommunicat(?:e|ing|es)\b", r"\btransmit(?:s|ting)?\b", r"\bsend(?:s|ing)?\b",
    r"\breceiv(?:e|es|ing)\b", r"\bsynchroniz(?:e|es|ing)?\b", r"\bupdate(?:s|ing)?\b",
    r"\bconnect(?:s|ing)?\b", r"\bcoupl(?:ed|ing|es)\b"
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


def _token_set(text: str) -> Set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _ngram_set(text: str, n: int) -> Set[str]:
    tokens = [t for t in re.findall(r"[a-z]+", text.lower())]
    return {" ".join(tokens[i:i+n]) for i in range(max(0, len(tokens) - n + 1))}


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
    unique_struct_nouns: Tuple[str, ...]
    claim_domain_tags: Tuple[str, ...]
    claim_domain_score: float
    claim_abstractness: float


@dataclass(frozen=True)
class ProductFeatures:
    token_count: int
    component_terms: int
    interaction_terms: int
    functional_terms: int
    functional_density: float
    unique_struct_nouns: Tuple[str, ...]
    product_domain_tags: Tuple[str, ...]
    product_domain_score: float


class FeatureBuilder:
    _nlp = None
    _embedder = None

    def __init__(self, spacy_model: str = "en_core_web_sm", st_model: str = "all-MiniLM-L6-v2"):
        if FeatureBuilder._nlp is None:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                FeatureBuilder._nlp = spacy.load(spacy_model, disable=["ner"])
        if FeatureBuilder._embedder is None:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                FeatureBuilder._embedder = SentenceTransformer(st_model)
        self.nlp = FeatureBuilder._nlp
        self.embedder = FeatureBuilder._embedder

    def _detect_domain(self, text: str) -> Tuple[Tuple[str, ...], float]:
        """Detect which domain(s) a text belongs to and return (domains, confidence)."""
        tokens = _token_set(text)
        domain_votes: Dict[str, float] = {d: 0.0 for d in DOMAINS}

        for domain, vocab in DOMAINS.items():
            # Components: highest weight
            comp_matches = len(tokens & set(vocab["components"]))
            # Actions: medium weight
            action_matches = len(tokens & set(vocab["actions"]))
            # Objects: lower weight
            obj_matches = len(tokens & set(vocab["objects"]))
            domain_votes[domain] = comp_matches * 3.0 + action_matches * 2.0 + obj_matches * 1.0

        total_votes = sum(domain_votes.values())
        if total_votes == 0:
            return (), 0.0

        sorted_domains = sorted(domain_votes.items(), key=lambda x: -x[1])
        top_domain, top_score = sorted_domains[0]
        if top_score == 0:
            return (), 0.0

        # If second-best is within 50% of top, text spans domains
        second_score = sorted_domains[1][1]
        if second_score > 0 and second_score >= top_score * 0.5:
            # Multi-domain text
            tied = [d for d, s in sorted_domains if s >= top_score * 0.5]
            return tuple(sorted(tied)), top_score / total_votes
        else:
            return (top_domain,), top_score / total_votes

    def _compute_abstractness(self, text: str, token_count: int) -> float:
        """Compute how abstract a text is (0=concrete, 1=very abstract)."""
        abstraction_count = _count_patterns(ABSTRACTION_PATTERNS, text)
        generic_count = text.lower().count("configured to") + text.lower().count("adapted to")
        return min(1.0, (abstraction_count + generic_count * 0.5) / max(1, token_count / 5))

    def build_claim_features(self, claim_text: str) -> ClaimFeatures:
        doc = self.nlp(claim_text)
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        words = [t for t in tokens if t.is_alpha or t.like_num]
        token_count = len(tokens)

        depths = _dependency_depths(doc)

        func_verbs = [t for t in tokens if t.pos_ == "VERB" and t.lemma_.lower() in FUNCTION_VERBS]
        struct_nouns = [t for t in tokens if t.pos_ in {"NOUN", "PROPN"} and t.lemma_.lower() in STRUCTURAL_NOUNS]

        func_density = (len(func_verbs) / token_count) if token_count else 0.0
        struct_density = (len(struct_nouns) / token_count) if token_count else 0.0

        abstraction_score = _count_patterns(ABSTRACTION_PATTERNS, claim_text)
        clause_markers = _count_markers(CLAUSE_MARKERS, claim_text)
        means_plus_function = 1 if re.search(r"\bmeans for\b", claim_text.lower()) else 0

        interaction_hits = sum(1 for p in INTERACTION_PATTERNS if re.search(p, claim_text.lower()))
        interaction_density = (interaction_hits / max(1, token_count)) * 10.0

        unique_nouns = tuple(sorted({t.lemma_.lower() for t in struct_nouns}))
        domain_tags, domain_score = self._detect_domain(claim_text)
        abstractness = self._compute_abstractness(claim_text, token_count)

        return ClaimFeatures(
            word_count=len(words), token_count=token_count,
            clause_markers=clause_markers,
            dep_depth_max=int(np.max(depths)),
            dep_depth_mean=float(np.mean(depths)),
            func_verb_density=float(func_density),
            struct_noun_density=float(struct_density),
            abstraction_score=int(abstraction_score),
            means_plus_function=int(means_plus_function),
            interaction_density=float(interaction_density),
            unique_struct_nouns=unique_nouns,
            claim_domain_tags=domain_tags,
            claim_domain_score=domain_score,
            claim_abstractness=abstractness,
        )

    def build_product_features(self, product_text: str) -> ProductFeatures:
        doc = self.nlp(product_text)
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        token_count = len(tokens)

        component_terms = sum(
            1 for t in tokens if t.pos_ in {"NOUN", "PROPN"} and t.lemma_.lower() in STRUCTURAL_NOUNS
        )
        interaction_terms = sum(
            1 for t in tokens
            if t.pos_ == "VERB" and re.search(
                r"(send|receive|transmit|sync|synchroniz|connect|communicat|update|"
                r"encrypt|decrypt|authenticate|verify|render|display|aggregate)",
                t.lemma_.lower()
            )
        )
        functional_terms = sum(
            1 for t in tokens if t.pos_ == "VERB" and t.lemma_.lower() in FUNCTION_VERBS
        )
        functional_density = (functional_terms / token_count) if token_count else 0.0

        struct_nouns = [t for t in tokens if t.pos_ in {"NOUN", "PROPN"} and t.lemma_.lower() in STRUCTURAL_NOUNS]
        unique_nouns = tuple(sorted({t.lemma_.lower() for t in struct_nouns}))
        domain_tags, domain_score = self._detect_domain(product_text)

        return ProductFeatures(
            token_count=token_count,
            component_terms=int(component_terms),
            interaction_terms=int(interaction_terms),
            functional_terms=int(functional_terms),
            functional_density=float(functional_density),
            unique_struct_nouns=unique_nouns,
            product_domain_tags=domain_tags,
            product_domain_score=domain_score,
        )

    def semantic_similarity(self, claim_text: str, product_text: str) -> float:
        a = self.embedder.encode([claim_text], normalize_embeddings=True)
        b = self.embedder.encode([product_text], normalize_embeddings=True)
        return float(cosine_similarity(a, b)[0][0])

    def build_feature_vector(self, claim_text: str, product_text: str) -> Dict[str, float]:
        c = self.build_claim_features(claim_text)
        p = self.build_product_features(product_text)
        s = self.semantic_similarity(claim_text, product_text)

        # ── Domain overlap ────────────────────────────────────────────────
        claim_domains = set(c.claim_domain_tags)
        prod_domains = set(p.product_domain_tags)
        domain_overlap = len(claim_domains & prod_domains)
        same_domain = 1.0 if claim_domains and prod_domains and claim_domains == prod_domains else 0.0

        # For multi-domain texts, check if any overlap
        any_domain_overlap = 1.0 if claim_domains and prod_domains and (claim_domains & prod_domains) else 0.0

        # ── Noun/verb structural overlap ──────────────────────────────────
        claim_nouns = set(c.unique_struct_nouns)
        prod_nouns = set(p.unique_struct_nouns)
        noun_jaccard = len(claim_nouns & prod_nouns) / max(1, len(claim_nouns | prod_nouns))
        noun_overlap_count = len(claim_nouns & prod_nouns)
        noun_overlap_claim = noun_overlap_count / max(1, len(claim_nouns))
        noun_overlap_prod = noun_overlap_count / max(1, len(prod_nouns))

        claim_all_nouns = set(t.lemma_.lower() for t in self.nlp(claim_text) if t.pos_ in {"NOUN", "PROPN"})
        prod_all_nouns = set(t.lemma_.lower() for t in self.nlp(product_text) if t.pos_ in {"NOUN", "PROPN"})
        all_noun_jaccard = len(claim_all_nouns & prod_all_nouns) / max(1, len(claim_all_nouns | prod_all_nouns))

        # ── N-gram lexical overlap ─────────────────────────────────────────
        bigram_jaccard = len(_ngram_set(claim_text, 2) & _ngram_set(product_text, 2)) / max(1, len(_ngram_set(claim_text, 2) | _ngram_set(product_text, 2)))
        trigram_jaccard = len(_ngram_set(claim_text, 3) & _ngram_set(product_text, 3)) / max(1, len(_ngram_set(claim_text, 3) | _ngram_set(product_text, 3)))

        # ── Length features ────────────────────────────────────────────────
        len_ratio = c.token_count / max(1, p.token_count)
        len_diff_norm = abs(c.token_count - p.token_count) / max(c.token_count, p.token_count, 1)

        # ── Abstraction alignment (key for scoring) ───────────────────────
        # Abstract claims (high abstractness) pair better with detailed products
        abs_alignment = 1.0 - abs(c.claim_abstractness - (1.0 - p.functional_density))
        abs_diff = abs(c.claim_abstractness - (1.0 - p.functional_density))

        # ── Domain mismatch penalty ──────────────────────────────────────
        # When both are detected but don't overlap = strong negative signal
        detected_both = 1.0 if (claim_domains and prod_domains) else 0.0
        domain_mismatch = 0.0
        if detected_both:
            domain_mismatch = 1.0 if claim_domains != prod_domains else 0.0
        # When neither is detected and both are short → suspicious (copy-paste of generic text)
        neither_detected = 1.0 if (not claim_domains and not prod_domains) else 0.0
        short_neither = neither_detected * (1.0 if c.token_count < 15 and p.token_count < 15 else 0.0)

        # ── Out-of-domain detection ──────────────────────────────────────
        tokens = _token_set(claim_text) | _token_set(product_text)
        ood_hits = len(tokens & OUT_OF_DOMAIN_KEYWORDS)
        ood_ratio = ood_hits / max(1, len(tokens))

        # ── Keyword density comparison ───────────────────────────────────
        # Claim should not be more abstract than product can cover
        claim_complexity = c.claim_abstractness * c.dep_depth_mean
        product_detail = p.functional_density * p.component_terms
        complexity_balance = claim_complexity - product_detail

        return {
            # Claim structure
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
            "claim_unique_nouns": float(len(c.unique_struct_nouns)),
            "claim_abstractness": float(c.claim_abstractness),
            "claim_domain_score": float(c.claim_domain_score),

            # Product structure
            "prod_token_count": float(p.token_count),
            "prod_component_terms": float(p.component_terms),
            "prod_interaction_terms": float(p.interaction_terms),
            "prod_functional_terms": float(p.functional_terms),
            "prod_functional_density": float(p.functional_density),
            "prod_unique_nouns": float(len(p.unique_struct_nouns)),
            "product_domain_score": float(p.product_domain_score),

            # Core similarity
            "semantic_similarity": float(s),

            # Domain overlap (strongest signal)
            "same_domain": float(same_domain),
            "domain_overlap_count": float(domain_overlap),
            "any_domain_overlap": float(any_domain_overlap),
            "domain_mismatch_penalty": float(domain_mismatch),
            "neither_domain_detected": float(neither_detected),
            "short_neither_penalty": float(short_neither),

            # Structural alignment
            "noun_jaccard": float(noun_jaccard),
            "noun_overlap_count": float(noun_overlap_count),
            "noun_overlap_claim": float(noun_overlap_claim),
            "noun_overlap_prod": float(noun_overlap_prod),
            "all_noun_jaccard": float(all_noun_jaccard),

            # Lexical
            "bigram_jaccard": float(bigram_jaccard),
            "trigram_jaccard": float(trigram_jaccard),

            # Length
            "len_ratio": float(len_ratio),
            "len_diff_norm": float(len_diff_norm),

            # Abstraction alignment
            "abs_alignment": float(abs_alignment),
            "abs_diff": float(abs_diff),

            # Complexity balance
            "complexity_balance": float(complexity_balance),
            "ood_keyword_ratio": float(ood_ratio),
        }