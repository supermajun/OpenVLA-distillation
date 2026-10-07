# Reproducing the experiment

## Scope and environment

The main study is a single-task pilot, not the full LIBERO benchmark. Training used an A100 80 GB with Python 3.11 and PyTorch 2.8.0/CUDA 12.8. Exported policy evaluation used Apple Silicon MPS with the environment recorded in `openvla_kd/requirements.lock.txt`. CUDA and MPS kernels need not produce identical floating-point values.

The repository preserves two project implementations to keep their experimental roles clear. No large model or dataset is included. Download the following public inputs according to their upstream terms; `openvla_kd/scripts/download_assets.py` pins their revisions:

| Input | Public source | Revision |
|---|---|---|
| Student | HuggingFaceTB/SmolVLM-500M-Instruct | `a7da5b986cb59b408707209984f360a5f4ad7e47` |
| Teacher | moojink/openvla-7b-oft-finetuned-libero-spatial | `6d0231af0e48c5985f1ff86908f4674b84bc049b` |
| Demonstrations | yifengzhu-hf/LIBERO-datasets | `f13aa24a3da8c43c7225569f28c562979fa0e35a` |
| Teacher code | moojink/openvla-oft | `e4287e94541f459edc4feabc4e181f537cd569a8` |
| Simulator tasks | Lifelong-Robot-Learning/LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |

## Prepare on Apple Silicon

Install `uv` before running the bootstrap. It creates the project-local environment, synchronizes the macOS lockfile and clones the pinned upstream code; it does not install software globally.

```bash
cd openvla_kd
python3 scripts/bootstrap.py
.venv/bin/python scripts/download_assets.py student
.venv/bin/python scripts/download_assets.py teacher
.venv/bin/python scripts/download_assets.py data
.venv/bin/python -m vla_kd.data \
  --hdf5 data/raw/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5 \
  --train-episodes 20 --val-episodes 5 --samples-per-episode 16 \
  --out data/expanded
.venv/bin/python -m vla_kd.teacher --device mps \
  --data data/expanded --out data/expanded_teacher_cache
.venv/bin/python -m pytest -q
```

The data builder takes the first 20 demonstrations for training and the last five for validation. Sixteen uniformly spaced observations per demonstration give 320/80 observations. Targets remain consecutive eight-action chunks from the original trajectory, with masked tail padding. This is not 320 independent demonstrations.

Teacher cache creation loads the 7B teacher and can require substantial memory, storage and time. The teacher is not loaded alongside student training. MPS and MuJoCo need access to the native macOS graphics environment; headless/sandbox restrictions may prevent rollout rendering even when CPU tests pass.

## Experimental training protocol

The CUDA research implementation is in `openvla_kd_cuda/vla_kd/`. It contains the student, objective, optimizer, data/cache validation and training logic. Infrastructure deployment and operation instructions are outside the scope of this public repository.

Each capacity starts from the shared pretrained initialization. Within a capacity, updates are cumulative (500 + 500 + 1,000), with model, master weights, optimizer and RNG state restored at stage boundaries. The experiment contains 6,000 total optimizer updates, not nine fresh independent training runs. It uses seed 17, microbatch one, accumulation one, LR 1e-4, weight decay 0.01, gradient clipping 1.0, action loss weight 1.0 and feature loss weight 0.1. Validation occurs every 50 updates and full checkpoints every 100.

Both packages expose `vla_kd`, so keep their environments and working directories separate. Checkpoint continuation requires compatible model, optimizer, preprocessing and cache definitions. Deployment-only policies do not include the Adam state needed for exact training continuation.

## Local policy evaluation

The local evaluator accepts a policy artifact and prepared validation data. With the local environment and public inputs prepared as above, run from `openvla_kd` and provide your own policy path:

```bash
.venv/bin/python -m vla_kd.evaluate \
  --policy /path/to/policy.pt --data data/expanded \
  --device mps --episodes 8 --initial-indices 0 1 2 3 4 5 6 7 \
  --max-steps 220 --execute-chunk 8 --seed 17 \
  --out runs/local_evaluation
```

Each reported policy was evaluated on initial states 0–7, seed 17, with a 220-action limit and an eight-action execution chunk. Offline L1 is averaged over 80 held-out demonstration observations. Warm forward timing uses batch one and two views, two warm-ups and ten measured calls. It excludes image preprocessing, simulator work and the complete control loop. Eight actions per forward call must not be interpreted as eight refreshed policy decisions.

## Inspect the released measurements

From the repository root, `python3 tools/verify_results.py` checks the nine records, CSV agreement, per-state success totals and cumulative training histories. This requires only the Python standard library and does not retrain or execute the simulator.

The public measurements are a selected export of the original experiment. Machine-specific metadata and all policy/optimizer weights are excluded.
