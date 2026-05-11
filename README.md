# LF RL PyTorch Baseline

This repository is the ente reinforcement-learning baseline for `aido-LF-sim-validation`.
It extends `duckietown/challenge-aido_lf-template-pytorch:ente` with the trained sample models and the RL solution entrypoint used for Duckiematrix evaluation.

Like the template, the baseline keeps its runtime in `solution/` and its shipped checkpoint in `models/`. The baseline-specific MVP lives in `training/`, which contains the RL trainer, evaluation script, and helper utilities. The older `duckietown_rl/` package, the legacy `node_launch.yaml`, and the redundant `learning/` namespace have been removed so the repository has one runtime surface and one baseline-owned training surface.

## Training

Run the baseline training flow from the repository root with:

- `python -m training.train`
- `python -m training.test`
