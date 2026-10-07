# Upstream research and dependencies

This project builds on the following independent projects. Their source, model weights and datasets are retrieved from upstream rather than vendored into this repository.

- [OpenVLA](https://github.com/openvla/openvla) — vision-language-action modeling.
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft) — teacher policy, continuous-action inference and robot preprocessing conventions.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — simulated task, initial states and demonstration data.
- [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) — pretrained student backbone.
- [Transformers](https://github.com/huggingface/transformers), [PyTorch](https://github.com/pytorch/pytorch), [robosuite](https://github.com/ARISE-Initiative/robosuite), and [MuJoCo](https://github.com/google-deepmind/mujoco) — runtime components.

See the pinned revisions in the reproduction guide and download/bootstrap scripts. Downloaded artifacts remain subject to their upstream licenses, model cards and dataset terms. This publication does not relicense upstream work, and no license for the project's original code has been selected here. Research use and redistribution permissions should be checked with the relevant rights holders.
