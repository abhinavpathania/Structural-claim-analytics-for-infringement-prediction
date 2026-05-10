import random
import csv
from pathlib import Path
import math

random.seed(42)


DOMAINS = {
    "data_sync": {
        "components": ["server", "client", "database", "processor", "memory", "network interface", "controller"],
        "actions": ["synchronize", "transmit", "receive", "update", "process", "store", "send"],
        "objects": ["record", "message", "event", "packet", "state update"]
    },
    "motion_alert": {
        "components": ["sensor", "camera", "processor", "controller", "wireless module", "memory"],
        "actions": ["detect", "generate", "transmit", "receive", "process", "store", "monitor"],
        "objects": ["motion", "frame", "alert", "signal", "event"]
    },
    "auth_security": {
        "components": ["processor", "memory", "secure module", "server", "client device", "interface"],
        "actions": ["authenticate", "encrypt", "decrypt", "store", "verify", "transmit"],
        "objects": ["key", "token", "credential", "request", "response"]
    },
    "power_hw": {
        "components": ["circuit", "voltage regulator", "power module", "sensor", "controller"],
        "actions": ["regulate", "monitor", "convert", "control", "process", "receive"],
        "objects": ["voltage", "current", "power state", "signal"]
    },
    "analytics_ui": {
        "components": ["dashboard", "application", "ui module", "report engine", "database"],
        "actions": ["render", "display", "generate", "aggregate", "process", "store"],
        "objects": ["chart", "report", "metric", "dashboard", "result"]
    }
}

# Out-of-domain vocabulary for true negatives (non-technical)
OUT_OF_DOMAIN_VOCAB = {
    "pharmaceutical": ["compound", "formulation", "carrier", "administer", "therapeutic", "ingredient", "solubility"],
    "construction": ["cement", "aggregate", "beam", "foundation", "structural", "load-bearing"],
    "ecommerce": ["cart", "checkout", "inventory", "shipment", "customer", "order", "pricing"],
    "finance": ["portfolio", "asset", "dividend", "equity", "bond", "hedge", "derivative"],
    "marketing": ["campaign", "conversion", "engagement", "demographic", "targeting", "impression"],
    "food": ["flavor", "ingredient", "preservative", "shelf-life", "calorie", "nutrient"],
}

ABSTRACT_PHRASES = ["configured to", "adapted to", "operable to", "arranged to", "operatively coupled to"]
STRUCTURAL_MARKERS = ["comprising", "including", "having", "with", "composed of", "including"]


def make_claim(domain_key: str, abstraction_level: float = 0.5) -> str:
    d = DOMAINS[domain_key]
    comp1 = random.choice(d["components"])
    comp2 = random.choice(d["components"])
    act1 = random.choice(d["actions"])
    act2 = random.choice(d["actions"])
    obj = random.choice(d["objects"])
    abs1 = random.choice(ABSTRACT_PHRASES)
    struct = random.choice(STRUCTURAL_MARKERS)

    if abstraction_level > 0.7:
        return (
            f"A system {struct} a {comp1} and a {comp2}, "
            f"the {comp1} {abs1} {act1} {obj}, "
            f"wherein the {comp2} {abs1} {act2} the {obj} based on predetermined parameters."
        )
    elif abstraction_level > 0.3:
        return (
            f"A system {struct} a {comp1} and a {comp2}, "
            f"the {comp1} configured to {act1} {obj} to a remote endpoint, "
            f"wherein the {comp2} {abs1} {act2} the {obj} and generate a response."
        )
    else:
        return (
            f"A system {struct} a {comp1} operably coupled to a {comp2}, "
            f"the {comp1} configured to {act1} {obj} over a network connection at a predetermined rate, "
            f"wherein the {comp2} {abs1} {act2} the {obj}, process the data, and transmit a response signal."
        )


def make_product_doc(domain_key: str, detail_level: float = 0.5) -> str:
    d = DOMAINS[domain_key]
    comps = random.sample(d["components"], k=min(3 + int(detail_level * 2), len(d["components"])))
    acts = random.sample(d["actions"], k=min(3, len(d["actions"])))
    obj = random.choice(d["objects"])

    if detail_level > 0.7:
        return (
            f"This product includes {', '.join(comps)}. "
            f"It can {acts[0]} and {acts[1]} {obj} in real-time. "
            f"The system supports {acts[2]} operations via REST API and WebSocket interfaces. "
            f"Communication happens over TCP/IP with encryption."
        )
    elif detail_level > 0.3:
        return (
            f"This product includes {', '.join(comps)}. "
            f"It can {acts[0]} and {acts[1]} {obj}. "
            f"The system also supports {acts[2]} operations via an interface."
        )
    else:
        return (
            f"This product features {', '.join(comps)}. "
            f"It supports {acts[0]} and {acts[1]} operations."
        )


def make_out_of_domain_claim(domain_key: str) -> str:
    """Generate a claim from a completely unrelated domain."""
    vocab = OUT_OF_DOMAIN_VOCAB[domain_key]
    comps = random.sample(vocab, k=2)
    acts = random.sample(vocab, k=2)
    objs = random.sample(vocab, k=2)
    struct = random.choice(STRUCTURAL_MARKERS)
    abs1 = random.choice(ABSTRACT_PHRASES)

    return (
        f"A composition {struct} {comps[0]} and {comps[1]}, "
        f"the {comps[0]} {abs1} {acts[0]} {objs[0]}, "
        f"wherein the composition is configured to {acts[1]} {objs[1]} over a sustained period."
    )


def make_out_of_domain_product(domain_key: str) -> str:
    """Generate a product description from an unrelated domain."""
    vocab = OUT_OF_DOMAIN_VOCAB[domain_key]
    comps = random.sample(vocab, k=3)
    acts = random.sample(vocab, k=2)
    objs = random.sample(vocab, k=2)

    return (
        f"This product contains {', '.join(comps)}. "
        f"It is designed to {acts[0]} and {acts[1]} {objs[0]}. "
        f"The formulation supports {objs[1]} under standard conditions."
    )


def _compute_similarity_score(claim_domain, product_domain,
                               claim_abs, prod_detail,
                               same_domain_override=None) -> float:
    if same_domain_override == "out_of_domain":
        return random.uniform(0.0, 12.0)

    if claim_domain == product_domain:
        # High similarity: abstract claims pair with detailed products
        if claim_abs > 0.5 and prod_detail > 0.5:
            # Well-aligned: abstract scope matches detailed implementation
            return random.uniform(70.0, 98.0)
        elif claim_abs > prod_detail:
            # Claim more abstract than product detail → partial coverage
            gap = claim_abs - prod_detail
            return random.uniform(45.0, 45.0 + (1 - gap) * 45.0)
        elif prod_detail > claim_abs:
            # Product more detailed than claim scope → product covers claim
            gap = prod_detail - claim_abs
            return random.uniform(50.0 + gap * 35.0, 90.0)
        else:
            # Roughly equal abstraction levels
            return random.uniform(55.0, 80.0)

    # Cross-domain: look up relatedness
    related_pairs = {
        ("data_sync", "auth_security"): (10, 40),
        ("motion_alert", "analytics_ui"): (5, 30),
        ("data_sync", "analytics_ui"): (5, 25),
        ("auth_security", "power_hw"): (5, 20),
        ("motion_alert", "power_hw"): (5, 20),
    }
    for (d1, d2), (lo, hi) in related_pairs.items():
        if (claim_domain == d1 and product_domain == d2) or (claim_domain == d2 and product_domain == d1):
            return random.uniform(lo, hi)

    return random.uniform(0.0, 18.0)


def generate_pairs(n_samples: int, out_path: Path, seed: int = 42):
    random.seed(seed)
    rows = []
    domain_keys = list(DOMAINS.keys())

    # Balanced scenario distribution:
    # 40% same domain (generates high-ish scores)
    # 25% cross-domain related (moderate scores)
    # 20% cross-domain unrelated (low scores)
    # 15% out-of-domain (very low scores — NEW)

    for i in range(n_samples):
        scenario = random.choices(
            ["same_domain", "cross_related", "cross_unrelated", "out_of_domain"],
            weights=[0.40, 0.25, 0.20, 0.15]
        )[0]

        if scenario == "same_domain":
            dom = random.choice(domain_keys)
            claim_abs = random.uniform(0.2, 0.9)
            prod_detail = random.uniform(0.2, 0.9)
            claim = make_claim(dom, claim_abs)
            product = make_product_doc(dom, prod_detail)
            label = _compute_similarity_score(dom, dom, claim_abs, prod_detail)

        elif scenario == "cross_related":
            dom_claim = random.choice(domain_keys)
            dom_prod = random.choice(domain_keys)
            while dom_prod == dom_claim:
                dom_prod = random.choice(domain_keys)
            claim_abs = random.uniform(0.3, 0.8)
            prod_detail = random.uniform(0.3, 0.8)
            claim = make_claim(dom_claim, claim_abs)
            product = make_product_doc(dom_prod, prod_detail)
            label = _compute_similarity_score(dom_claim, dom_prod, claim_abs, prod_detail)

        elif scenario == "cross_unrelated":
            dom_claim = random.choice(domain_keys)
            dom_prod = random.choice(domain_keys)
            while dom_prod == dom_claim:
                dom_prod = random.choice(domain_keys)
            claim = make_claim(dom_claim, random.uniform(0.3, 0.8))
            product = make_product_doc(dom_prod, random.uniform(0.3, 0.8))
            label = _compute_similarity_score(dom_claim, dom_prod, 0.5, 0.5)

        else:  # out_of_domain
            ood_key = random.choice(list(OUT_OF_DOMAIN_VOCAB.keys()))
            dom_claim = random.choice(domain_keys)
            claim = make_claim(dom_claim, random.uniform(0.4, 0.8))
            product = make_out_of_domain_product(ood_key)
            label = _compute_similarity_score(None, None, 0.5, 0.5, "out_of_domain")

        rows.append({
            "pair_id": f"pair_{i:05d}",
            "claim_text": claim,
            "product_text": product,
            "label": round(label, 2)
        })

    random.shuffle(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "claim_text", "product_text", "label"])
        w.writeheader()
        w.writerows(rows)

    labels = [r["label"] for r in rows]
    print(f"Generated {n_samples} pairs with label distribution:")
    print(f"  0-20%:   {sum(1 for l in labels if l <= 20)}")
    print(f"  21-50%:  {sum(1 for l in labels if 20 < l <= 50)}")
    print(f"  51-75%:  {sum(1 for l in labels if 50 < l <= 75)}")
    print(f"  76-100%: {sum(1 for l in labels if l > 75)}")
    mean_l = sum(labels) / len(labels)
    std_l = math.sqrt(sum((l - mean_l) ** 2 for l in labels) / len(labels))
    print(f"  Mean: {mean_l:.2f}, Std: {std_l:.2f}")


if __name__ == "__main__":
    generate_pairs(
        n_samples=5000,
        out_path=Path("data/synthetic_pairs_5k.csv")
    )
    print("Saved: data/synthetic_pairs_5k.csv")