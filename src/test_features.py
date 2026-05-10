from __future__ import annotations
from .feature_builder import FeatureBuilder

fb = FeatureBuilder()

# ---------------------------------------------------------------------------
# Test cases: (description, claim_text, product_text, expected_risk_range)
# expected_risk_range is a hint for human review — not an automated assertion
# ---------------------------------------------------------------------------

TEST_CASES = [
    # ── High risk / direct infringement ────────────────────────────────────
    (
        "Direct structural match",
        "1. A system comprising: a processor configured to receive data from a sensor, "
        "transmit processed data to a remote server, and authenticate user credentials.",
        "A processing unit that receives sensor input, authenticates users, and sends data "
        "to a cloud server.",
        (70, 100),
    ),
    (
        "Identical functional verbs",
        "A method for transmitting encrypted data between a client and server while "
        "authenticating the user session.",
        "Device that encrypts and transmits data between client and server while "
        "authenticating user sessions.",
        (65, 95),
    ),
    (
        "Complete claim scope covered",
        "An apparatus having a memory, a processor, an interface, and a network bus, "
        "wherein the processor is configured to execute instructions stored in the memory "
        "to synchronize data with an external database.",
        "Electronic device with memory, processor, interface, and bus that synchronizes "
        "data with external databases.",
        (60, 90),
    ),
    (
        "Multiple structural nouns + interaction",
        "A controller comprising: a processor, a memory module, a communication interface, "
        "and a network engine, wherein the controller is configured to receive signals, "
        "process received signals, and transmit processed signals to a remote server.",
        "Controller device with processor, memory, communication interface, and network "
        "engine that receives, processes, and transmits signals to a server.",
        (65, 95),
    ),

    # ── Medium risk / partial overlap ────────────────────────────────────
    (
        "Some structural matches, different function",
        "A system for processing audio signals comprising a processor, a memory, and an "
        "output interface configured to transmit processed audio to a display device.",
        "Video processor with memory and display output.",
        (30, 60),
    ),
    (
        "Shared components, different domain",
        "An apparatus for wireless data transmission comprising a transmitter, a receiver, "
        "and an antenna configured to communicate with a remote server.",
        "Radio transmitter with antenna and receiver for short-range communication.",
        (25, 55),
    ),
    (
        "Overlapping abstractions",
        "A device adapted to receive input from a sensor and configured to process the "
        "input to generate output data for transmission to an external system.",
        "Sensor input module that processes signals and generates output data.",
        (35, 65),
    ),
    (
        "Partial clause structure overlap",
        "A method comprising: receiving data, processing received data based on a "
        "predetermined algorithm, and transmitting processed data to a remote server.",
        "System that receives and processes data according to a defined algorithm.",
        (30, 60),
    ),

    # ── Low risk / minimal overlap ────────────────────────────────────────
    (
        "Unrelated domain",
        "A pharmaceutical composition comprising an active ingredient and a carrier, "
        "wherein the composition is configured to release the active ingredient over time.",
        "A data processing system for managing customer records.",
        (0, 25),
    ),
    (
        "Different function verbs",
        "A system configured to receive patient health data and transmit diagnostic "
        "recommendations to a healthcare provider.",
        "A document editor for writing and formatting text documents.",
        (0, 20),
    ),
    (
        "Generic product description",
        "An apparatus for transmitting data comprising a processor and a memory unit "
        "configured to store and forward data packets.",
        "A computer.",
        (0, 30),
    ),
    (
        "No structural noun overlap",
        "A method for rendering three-dimensional graphics on a display device, "
        "comprising a rendering engine and a frame buffer.",
        "Text formatting software for plain text documents.",
        (0, 20),
    ),
    (
        "Completely unrelated",
        "A cement composition comprising portland cement, aggregate, and water, wherein "
        "the composition hardens to form a structural element.",
        "An e-commerce platform for selling clothing.",
        (0, 15),
    ),

    # ── Edge cases / tricky language ──────────────────────────────────────
    (
        "Means-plus-function claim",
        "1. A circuit for processing signals, comprising: means for receiving a signal; "
        "means for processing the received signal; and means for transmitting the processed "
        "signal to an output.",
        "Signal processing circuit with input, processing, and output stages.",
        (50, 80),
    ),
    (
        "Multiple abstraction patterns",
        "A device configured to be operable to receive input data, adapted to process "
        "the input data, and arranged to transmit output to a remote system.",
        "Data processing device that receives, processes, and outputs data.",
        (40, 70),
    ),
    (
        "Long claim, short product",
        "A system for managing inventory comprising: a database server connected to a "
        "network; a processor configured to execute instructions stored in the database "
        "server; an interface module adapted to receive inventory data; and a communication "
        "engine configured to transmit inventory updates to remote clients via the network.",
        "Inventory management software.",
        (20, 50),
    ),
    (
        "Short claim, long product",
        "A processor configured to transmit data to a server.",
        "High-performance CPU with advanced architecture that executes instructions, manages "
        "memory operations, processes multiple threads simultaneously, and transmits processed "
        "data and results to connected servers and cloud infrastructure via high-bandwidth "
        "network interfaces.",
        (30, 60),
    ),
    (
        "Claim with no function verbs",
        "A composition comprising a first component, a second component, and a third "
        "component in a ratio of approximately 1:2:3.",
        "Chemical mixture with three ingredients in 1:2:3 proportion.",
        (10, 40),
    ),
    (
        "Claim with no structural nouns",
        "A method comprising performing a first action and a second action.",
        "System that performs multiple sequential actions.",
        (10, 40),
    ),
    (
        "Clause markers in claim",
        "A system wherein the processor is configured to receive data, such that the "
        "received data is processed, and whereby the processed data is transmitted, "
        "based on a predetermined condition.",
        "Processor that receives, processes, and transmits data conditionally.",
        (40, 70),
    ),

    # ── Boundary / mixed signal ───────────────────────────────────────────
    (
        "Same structural nouns but different verbs",
        "A memory module comprising a storage unit and a controller, configured to "
        "store data and transmit stored data to an external device.",
        "A storage device having a storage unit and a controller that receives and "
        "erases data from the storage unit.",
        (20, 50),
    ),
    (
        "Same verbs but different structural nouns",
        "A system comprising a server, a database, and a network interface configured "
        "to receive and transmit data.",
        "A device comprising a chip, a screen, and a keypad that receives and transmits signals.",
        (20, 50),
    ),
    (
        "High semantic similarity, no structural overlap",
        "A method for establishing a secure communication channel between a client and "
        "a server by performing mutual authentication and key exchange.",
        "An approach to setting up an encrypted connection by exchanging credentials and "
        "keys between endpoints.",
        (30, 60),
    ),
]


def run_tests():
    print(f"{'#':<3} {'Description':<48} {'Sim':>6}  {'Label':>9}")
    print("-" * 70)
    for i, (desc, claim, product, _) in enumerate(TEST_CASES, 1):
        feats = fb.build_feature_vector(claim, product)
        sim = feats["semantic_similarity"] * 100
        print(f"{i:<3} {desc:<48} {sim:>5.1f}%")

    print(f"\n{len(TEST_CASES)} test cases loaded.")
    print("Run prediction on specific cases with: python -m src.predict --claim '...' --product '...'")


if __name__ == "__main__":
    run_tests()