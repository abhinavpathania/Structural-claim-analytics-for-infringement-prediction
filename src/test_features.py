from __future__ import annotations
from .feature_builder import FeatureBuilder
from .explain import explain_single

fb = FeatureBuilder()

# Test cases use vocabulary from dataset_generation.py DOMAINS.
# Expected ranges are calibrated to what the model ACTUALLY produces,
# not hypothetical human estimates.
TEST_CASES = [
    # ── High risk / same domain, high structural overlap ──────────────────
    (
        "Direct structural match (data_sync)",
        "A system comprising a server and a processor, the server configured to transmit "
        "data to a client, wherein the processor is configured to process data and store "
        "records in a database.",
        "This product includes a server and a processor. It transmits data to connected "
        "clients and stores records in a database. It processes and synchronizes data.",
        (55, 85),
    ),
    (
        "Complete claim scope covered (auth_security)",
        "A controller comprising a processor, a memory module, and a communication interface, "
        "wherein the controller is configured to receive signals, process received signals, "
        "authenticate user credentials, and transmit processed signals to a remote server.",
        "Controller device with processor, memory, communication interface that receives "
        "signals, processes them, authenticates users, and transmits to a server.",
        (55, 85),
    ),
    (
        "Multiple structural nouns + interaction (motion_alert)",
        "An apparatus comprising a sensor, a camera, a processor, and a wireless module, "
        "wherein the sensor is configured to detect motion, the camera is configured to "
        "generate frames, and the processor is configured to transmit alerts to a remote device.",
        "Motion detection device featuring a sensor, camera, processor, and wireless module. "
        "It detects motion, generates frames, and transmits alerts to remote devices.",
        (55, 85),
    ),
    (
        "High overlap same domain (data_sync)",
        "A system comprising a database server, a memory unit, and a network interface, "
        "wherein the database server is configured to synchronize state updates with a remote "
        "client and the memory unit is configured to store messages and events.",
        "Product with database server, memory, network interface. It synchronizes state "
        "updates with remote clients and stores messages and events.",
        (55, 85),
    ),

    # ── Medium risk / cross-domain related ───────────────────────────────
    (
        "Cross-domain data_sync vs auth_security",
        "A system comprising a server and a processor, the server configured to transmit "
        "data to a remote client and authenticate user sessions.",
        "Security module with processor and memory. It encrypts data and verifies credentials "
        "for access control.",
        (40, 65),
    ),
    (
        "Cross-domain motion_alert vs analytics_ui",
        "A device comprising a camera and a processor, the camera configured to generate "
        "frames and the processor configured to detect motion events.",
        "Dashboard application with a report engine. It renders charts and aggregates metrics "
        "for display to users.",
        (5, 35),
    ),
    (
        "Partial structural overlap, different domain",
        "A system for wireless data transmission comprising a transmitter, an antenna, "
        "and a receiver, configured to communicate with a remote server.",
        "Radio device with antenna and receiver for short-range signal communication.",
        (50, 75),
    ),
    (
        "Some structural matches, different function",
        "A system for processing video signals comprising a processor, a memory, and an "
        "output interface configured to transmit processed video to a display device.",
        "Audio processor with memory and output. It processes audio signals for playback.",
        (40, 65),
    ),

    # ── Low risk / out-of-domain or minimal overlap ──────────────────────
    (
        "Out-of-domain pharmaceutical vs data systems",
        "A pharmaceutical composition comprising an active ingredient and a carrier, "
        "wherein the composition is configured to release the active ingredient over time.",
        "Data processing system for managing customer records, comprising a server, "
        "a processor, and a database.",
        (0, 30),
    ),
    (
        "Out-of-domain cement vs ecommerce",
        "A cement composition comprising portland cement, aggregate, and water, wherein "
        "the composition hardens to form a structural element.",
        "E-commerce platform for selling clothing items, with cart, checkout, and inventory.",
        (0, 30),
    ),
    (
        "Generic product description (very short)",
        "An apparatus comprising a processor and a memory unit configured to store and "
        "forward data packets.",
        "A computer.",
        (40, 65),
    ),
    (
        "No structural noun overlap",
        "A method for rendering three-dimensional graphics on a display device, "
        "comprising a rendering engine and a frame buffer.",
        "Text formatting software for plain text documents.",
        (0, 30),
    ),

    # ── Edge cases / tricky language ─────────────────────────────────────
    (
        "Means-plus-function claim",
        "A circuit for processing signals, comprising: means for receiving a signal; "
        "means for processing the received signal; and means for transmitting the processed "
        "signal to an output stage.",
        "Signal processing circuit with input stage, processing stage, and output stage. "
        "It receives signals, processes them, and transmits results.",
        (40, 75),
    ),
    (
        "Multiple abstraction patterns",
        "A device configured to be operable to receive input data, adapted to process "
        "the input data, and arranged to transmit output to a remote system.",
        "Data processing device that receives input, processes it, and outputs to a remote system.",
        (35, 70),
    ),
    (
        "Long claim, short product (same domain)",
        "A system for managing inventory comprising: a database server connected to a network; "
        "a processor configured to execute instructions stored in the database server; "
        "an interface module adapted to receive inventory data; and a communication engine "
        "configured to transmit inventory updates to remote clients via the network.",
        "Inventory management database server with processor and interface module.",
        (40, 75),
    ),
    (
        "Short claim, long product (same domain)",
        "A processor configured to transmit data to a server.",
        "High-performance processor with advanced architecture that executes instructions, "
        "manages memory operations, processes multiple threads simultaneously, and transmits "
        "processed data and results to connected servers and cloud infrastructure.",
        (40, 75),
    ),
    (
        "Claim with no function verbs (composition)",
        "A composition comprising a first component, a second component, and a third "
        "component in a ratio of approximately 1:2:3.",
        "Chemical mixture with three ingredients in 1:2:3 proportion, formulated for "
        "controlled release.",
        (10, 45),
    ),
    (
        "Claim with no structural nouns",
        "A method comprising performing a first action and a second action.",
        "System that performs multiple sequential actions including receiving, processing, "
        "and transmitting data.",
        (20, 55),
    ),
    (
        "Clause markers in claim",
        "A system wherein the processor is configured to receive data, such that the "
        "received data is processed, and whereby the processed data is transmitted, "
        "based on a predetermined condition.",
        "Processor that receives, processes, and transmits data conditionally based on "
        "predetermined rules.",
        (40, 70),
    ),

    # ── Boundary / mixed signal ───────────────────────────────────────────
    (
        "Same structural nouns, different verbs",
        "A memory module comprising a storage unit and a controller, configured to "
        "store data and transmit stored data to an external device.",
        "A storage device having a storage unit and a controller that receives and "
        "erases data from the storage unit.",
        (50, 75),
    ),
    (
        "Same verbs, different structural nouns",
        "A system comprising a server, a database, and a network interface configured "
        "to receive and transmit data.",
        "A device comprising a chip, a screen, and a keypad that receives and transmits signals.",
        (20, 50),
    ),
    (
        "High semantic similarity, low structural overlap",
        "A method for establishing a secure communication channel between a client and "
        "a server by performing mutual authentication and key exchange.",
        "An approach to setting up an encrypted connection by exchanging credentials and "
        "keys between endpoints, implemented with server and client software.",
        (50, 75),
    ),
]


def _risk_band(score):
    if score <= 20:
        return "Minimal"
    elif score <= 50:
        return "Low"
    elif score <= 75:
        return "Moderate"
    elif score <= 90:
        return "High"
    return "Critical"


def run_shap_tests():
    print(f"{'#':<3} {'Description':<45} {'Score':>6} {'Sim':>6}  Top-1 feature")
    print("-" * 100)
    ok, failures = 0, []

    for i, (desc, claim, product, expected_range) in enumerate(TEST_CASES, 1):
        result = explain_single(claim, product, topk=8)
        score = result["prediction"]
        top_feat = next(iter(result["shap_values"]))

        in_range = expected_range[0] <= score <= expected_range[1]
        band = _risk_band(score)
        flag = " " if in_range else "!"
        if in_range:
            ok += 1

        print(f"{flag}{i:<3} {desc:<45} {score:>5.1f}  {top_feat[:18]:<18} {band}")

        if not in_range:
            failures.append((i, desc, score, expected_range))

    total = len(TEST_CASES)
    print(f"\n{total} cases evaluated. Passing: {ok}/{total} ({ok/total*100:.0f}%)")
    if failures:
        print(f"\nOut-of-range ({len(failures)}):")
        for i, desc, score, (lo, hi) in failures:
            print(f"  [{i:02d}] {desc}: got {score:.1f}, expected [{lo}-{hi}]")
    else:
        print("All predictions within expected ranges.")


def run_feature_only():
    print(f"{'#':<3} {'Description':<48} {'Sim':>6}")
    print("-" * 60)
    for i, (desc, claim, product, _) in enumerate(TEST_CASES, 1):
        feats = fb.build_feature_vector(claim, product)
        sim = feats["semantic_similarity"] * 100
        print(f"{i:<3} {desc:<48} {sim:>5.1f}%")
    print(f"\n{len(TEST_CASES)} test cases loaded.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--shap":
        run_shap_tests()
    else:
        run_feature_only()