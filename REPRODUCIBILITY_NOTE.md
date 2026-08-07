# Reproducibility Note

## Current status

The reported results use the released reference checkpoint `hetero_model.pt`.
The training script `train_hetero.py` is runnable and uses the released expert
states, but a fresh retraining does **not** reproduce the reference checkpoint.

## Actual retraining comparison

We ran `train_hetero.py` with the fixed settings in the script (30 epochs,
batch 64, Adam lr=1e-3, seed 42). This produced `hetero_model_retrained.pt`.
Both checkpoints were then evaluated on the same 49 official JSPLIB instances
with `run_corrected_official.py`.

| Model | Wins vs SPT | Mean improvement vs SPT | Mean known-best gap |
|---|---|---|---|
| Reference `hetero_model.pt` | 44/49 | +12.6% | 56.9% |
| Retrained `hetero_model_retrained.pt` | 25/49 | -1.5% | 83.0% |

Per-instance makespans match on 0 of 49 instances.

Raw retrained results: `results_official_retrained_49.json`.

## Implication for the manuscript

The paper should not claim that `train_hetero.py` reproduces the reported
numbers. The correct statement is:

> The released checkpoint is the authoritative artifact for all reported
> results. The training script and expert data reproduce the training pipeline,
> but not necessarily the exact trained model.

This limitation is already reflected in the README and Data Availability
section.
