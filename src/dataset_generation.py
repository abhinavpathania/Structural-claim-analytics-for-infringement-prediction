import random
import csv
from pathlib import Path
import math

random.seed(42)

DOMAINS = {
    "data_sync": {
        "components": ["server", "client", "database", "processor", "memory", "network interface", "controller"],
        "actions": ["synchronize", "transmit", "receive", "update", "process", "store"],
        "objects": ["records", "messages", "events", "packets", "state updates"]
    },
    "motion_alert": {
        "components": ["sensor", "camera", "processor", "controller", "wireless module", "memory"],
        "actions": ["detect", "generate", "transmit", "receive", "process", "store"],
        "objects": ["motion", "frames", "alerts", "signals", "events"]
    },
    "auth_security": {
        "components": ["processor", "memory", "secure module", "server", "client device", "interface"],
        "actions": ["authenticate", "encrypt", "decrypt", "store", "verify", "transmit"],
        "objects": ["keys", "tokens", "credentials", "requests", "responses"]
    },
    "power_hw": {
        "components": ["circuit", "voltage regulator", "power module", "sensor", "controller"],
        "actions": ["regulate", "monitor", "convert", "control"],
        "objects": ["voltage", "current", "power state"]
    },
    "analytics_ui": {
        "components": ["dashboard", "application", "ui module", "report engine", "database"],
        "actions": ["render", "display", "generate", "aggregate"],
        "objects": ["charts", "reports", "metrics", "dashboards"]
    }
}

ABSTRACT_PHRASES = ["configured to", "adapted to", "operable to", "arranged to"]
STRUCTURAL_MARKERS = ["comprising", "including", "having", "with", "composed of"]


def make_claim(domain_key: str, abstraction_level: float = 0.5) -> str:
    """Generate claim with controlled abstraction level (0-1)."""
    d = DOMAINS[domain_key]
    comp1 = random.choice(d["components"])
    comp2 = random.choice(d["components"])
    act1 = random.choice(d["actions"])
    act2 = random.choice(d["actions"])
    obj = random.choice(d["objects"])
    abs1 = random.choice(ABSTRACT_PHRASES)
    struct = random.choice(STRUCTURAL_MARKERS)

    if abstraction_level > 0.7:
        # Very abstract: use "configured to" language
        return (
            f"A system {struct} a {comp1} and a {comp2}, "
            f"the {comp1} {abs1} {act1} {obj}, "
            f"wherein the {comp2} {abs1} {act2} the {obj} based on predetermined parameters."
        )
    elif abstraction_level > 0.3:
        # Moderate abstraction
        return (
            f"A system {struct} a {comp1} and a {comp2}, "
            f"the {comp1} configured to {act1} {obj} to a remote endpoint, "
            f"wherein the {comp2} {abs1} {act2} the {obj} and generate a response."
        )
    else:
        # Concrete: specific technical details
        return (
            f"A system {struct} a {comp1} operably coupled to a {comp2}, "
            f"the {comp1} configured to {act1} {obj} over a network connection at a predetermined rate, "
            f"wherein the {comp2} {abs1} {act2} the {obj}, process the data, and transmit a response signal."
        )


def make_product_doc(domain_key: str, detail_level: float = 0.5) -> str:
    """Generate product doc with controlled detail level (0-1)."""
    d = DOMAINS[domain_key]
    comps = random.sample(d["components"], k=min(3 + int(detail_level * 2), len(d["components"])))
    acts = random.sample(d["actions"], k=min(3, len(d["actions"])))
    obj = random.choice(d["objects"])

    if detail_level > 0.7:
        # High detail: specific implementation
        return (
            f"This product includes {', '.join(comps)}. "
            f"It can {acts[0]} and {acts[1]} {obj} in real-time. "
            f"The system supports {acts[2]} operations via REST API and WebSocket interfaces. "
            f"Communication happens over TCP/IP with encryption."
        )
    elif detail_level > 0.3:
        # Moderate detail
        return (
            f"This product includes {', '.join(comps)}. "
            f"It can {acts[0]} and {acts[1]} {obj}. "
            f"The system also supports {acts[2]} operations via an interface."
        )
    else:
        # Minimal detail
        return (
            f"This product features {', '.join(comps)}. "
            f"It supports {acts[0]} and {acts[1]} operations."
        )


def _compute_similarity_score(claim_domain: str, product_domain: str,
                               claim_abs: float, prod_detail: float) -> float:
    """
    Compute a realistic infringement percentage (0-100) based on:
    - Domain match (higher = more similar)
    - Abstraction/detail alignment (claims + products both abstract = higher match)
    - Feature overlap potential
    """
    if claim_domain == product_domain:
        base_score = random.uniform(45, 95)  # Same domain: moderate to very high
    else:
        # Cross-domain: add some overlap occasionally (related tech areas)
        related_cross = {
            ("data_sync", "auth_security"): random.uniform(10, 35),
            ("motion_alert", "analytics_ui"): random.uniform(5, 25),
        }
        for (d1, d2), score in related_cross.items():
            if (claim_domain == d1 and product_domain == d2) or \
               (claim_domain == d2 and product_domain == d1):
                base_score = score
                break
        else:
            base_score = random.uniform(0, 15)  # Unrelated domains

    # Adjust based on abstraction-detail alignment
    # Abstract claims should match with detailed products better (scope coverage)
    alignment_bonus = 1 - abs(claim_abs - prod_detail) * 30
    final_score = base_score + alignment_bonus

    return max(0.0, min(100.0, final_score))


def generate_pairs(n_samples: int, out_path: Path, seed: int = 42):
    """
    Generate synthetic claim-product pairs with continuous percentage labels.

    Labels are infringement scores from 0-100 representing:
    - 0-20%: No/minimal infringement
    - 21-50%: Low infringement
    - 51-75%: Moderate infringement
    - 76-90%: High infringement
    - 91-100%: Very high/identical infringement
    """
    random.seed(seed)
    rows = []
    domain_keys = list(DOMAINS.keys())

    for i in range(n_samples):
        # Determine scenario type
        scenario = random.choices(
            ["same_domain", "cross_domain", "partial_overlap"],
            weights=[0.5, 0.35, 0.15]
        )[0]

        if scenario == "same_domain":
            dom = random.choice(domain_keys)
            claim_abs = random.uniform(0.2, 0.9)
            prod_detail = random.uniform(0.2, 0.9)
            claim = make_claim(dom, claim_abs)
            product = make_product_doc(dom, prod_detail)
            label = _compute_similarity_score(dom, dom, claim_abs, prod_detail)

        elif scenario == "cross_domain":
            dom_claim = random.choice(domain_keys)
            dom_prod = random.choice(domain_keys)
            claim_abs = random.uniform(0.3, 0.8)
            prod_detail = random.uniform(0.3, 0.8)
            claim = make_claim(dom_claim, claim_abs)
            product = make_product_doc(dom_prod, prod_detail)
            label = _compute_similarity_score(dom_claim, dom_prod, claim_abs, prod_detail)

        else:  # partial_overlap
            # Pick related domains with partial overlap
            dom_claim = "data_sync"
            dom_prod = random.choice(["data_sync", "auth_security"])
            claim_abs = random.uniform(0.4, 0.8)
            prod_detail = random.uniform(0.4, 0.8)
            claim = make_claim(dom_claim, claim_abs)
            product = make_product_doc(dom_prod, prod_detail)
            label = _compute_similarity_score(dom_claim, dom_prod, claim_abs, prod_detail)

        rows.append({
            "pair_id": f"pair_{i:05d}",
            "claim_text": claim,
            "product_text": product,
            "label": round(label, 2)
        })

    # Shuffle the rows
    random.shuffle(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "claim_text", "product_text", "label"])
        w.writeheader()
        w.writerows(rows)

    # Distribution summary
    labels = [r["label"] for r in rows]
    print(f"Generated {n_samples} pairs with label distribution:")
    print(f"  0-20%: {sum(1 for l in labels if l <= 20)}")
    print(f"  21-50%: {sum(1 for l in labels if 20 < l <= 50)}")
    print(f"  51-75%: {sum(1 for l in labels if 50 < l <= 75)}")
    print(f"  76-100%: {sum(1 for l in labels if l > 75)}")
    print(f"  Mean: {sum(labels)/len(labels):.2f}, Std: {math.sqrt(sum((l-sum(labels)/len(labels))**2 for l in labels)/len(labels)):.2f}")


if __name__ == "__main__":
    generate_pairs(
        n_samples=5000,
        out_path=Path("data/synthetic_pairs_5k.csv")
    )
    print("Saved: data/synthetic_pairs_5k.csv")


