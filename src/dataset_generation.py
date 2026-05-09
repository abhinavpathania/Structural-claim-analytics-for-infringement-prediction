import random
import csv
from pathlib import Path

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

def make_claim(domain_key: str) -> str:
    d = DOMAINS[domain_key]
    comp1 = random.choice(d["components"])
    comp2 = random.choice(d["components"])
    act1 = random.choice(d["actions"])
    act2 = random.choice(d["actions"])
    obj = random.choice(d["objects"])
    abs1 = random.choice(ABSTRACT_PHRASES)

    # independent-claim style
    return (
        f"A system comprising a {comp1} and a {comp2}, "
        f"the {comp1} {abs1} {act1} {obj} to a remote endpoint, "
        f"wherein the {comp2} is {abs1} {act2} the {obj} and generate a response."
    )

def make_product_doc(domain_key: str) -> str:
    d = DOMAINS[domain_key]
    comps = random.sample(d["components"], k=min(3, len(d["components"])))
    acts = random.sample(d["actions"], k=min(3, len(d["actions"])))
    obj = random.choice(d["objects"])
    return (
        f"This product includes {', '.join(comps)}. "
        f"It can {acts[0]} and {acts[1]} {obj}. "
        f"The system also supports {acts[2]} operations via an interface."
    )

def generate_pairs(n_pos: int, n_neg: int, out_path: Path):
    rows = []
    domain_keys = list(DOMAINS.keys())

    # positive: same domain
    for i in range(n_pos):
        dom = random.choice(["data_sync", "motion_alert", "auth_security"])
        claim = make_claim(dom)
        product = make_product_doc(dom)
        rows.append((f"pos_{i}", claim, product, 1))

    # negative: mismatched domains
    for i in range(n_neg):
        dom_claim = random.choice(["data_sync", "motion_alert", "auth_security"])
        dom_prod = random.choice(["power_hw", "analytics_ui"])
        claim = make_claim(dom_claim)
        product = make_product_doc(dom_prod)
        rows.append((f"neg_{i}", claim, product, 0))

    random.shuffle(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "claim_text", "product_text", "label"])
        w.writerows(rows)

if __name__ == "__main__":
    generate_pairs(
        n_pos=2500,
        n_neg=2500,
        out_path=Path("data/synthetic_pairs_5k.csv")
    )
    print("Saved: data/synthetic_pairs_5k.csv")


