# Local data preparation and evaluation

This is the Apple Silicon implementation used to prepare the teacher cache, run early pilot experiments and evaluate exported student policies in LIBERO. The CUDA research implementation lives in the sibling `openvla_kd_cuda` directory.

Follow the [reproduction guide](../docs/REPRODUCING.md). `scripts/bootstrap.py` creates a local Python 3.11 environment and retrieves pinned upstream repositories. `requirements.lock.txt` records the original macOS environment; it is not a Linux lockfile.

Run tests from this directory using an environment with the declared dependencies:

```bash
.venv/bin/python -m pytest -q
```

The `run_pilot.py`, `run_followup.py` and `run_capacity2b.py` scripts preserve historical experiments. They assume the required assets and, for some follow-ups, prior runs already exist. They are separate from the reported nine-checkpoint experiment. The default local training optimizer is BF16; the main reported experiment explicitly uses FP32 master weights and states. Do not treat results from these different stages as a controlled optimizer-only comparison.
