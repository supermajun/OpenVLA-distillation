# CUDA research implementation

This package retains the scientific implementation used to train the 500M, 1B and 2B students: the model, objective, teacher-cache interface, BF16 parameters/gradients, FP32 master weights and AdamW states. Deterministic visual pooling uses explicit adaptive-bin means.

The public package contains no infrastructure deployment or operation workflow. The [reproduction guide](../docs/REPRODUCING.md) describes public inputs, the experimental protocol and local evaluation.

The retained tests cover model behavior, loss computation, optimizer precision, data validation, deterministic pooling and checkpoint correctness. Run them from this directory with an environment containing the declared dependencies:

```bash
python -m pytest -q
```

Both this package and the sibling `openvla_kd` package expose `vla_kd`; use separate environments/working directories. CUDA-only tests require an NVIDIA GPU. CPU test results do not establish full-size CUDA reproduction.
