# Research implementation validation

The public source was checked on 7 October 2026 without rerunning full training or simulator experiments.

The retained local implementation and CUDA research implementation have separate test suites. These cover data/cache validation, model expansion, loss computation, optimizer precision, deterministic visual pooling, report generation and checkpoint interruption/resume. CUDA-specific cases require an NVIDIA GPU and are skipped on the local Mac.

`python3 tools/verify_results.py` verifies all nine published records, CSV agreement, 72 per-state outcomes and 6,000 cumulative update records. The released L1 values and success outcomes were checked against the original evaluation files.

The public package preserves research code and selected numeric evidence. Infrastructure deployment and operation workflows are excluded. Downloaded upstream dependencies, simulator assets, trained policies and teacher caches are also outside this repository. Fresh installation and full reproduction require those external inputs and were not repeated for this check.
