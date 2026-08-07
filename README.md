# Teacher-Distilled Lightweight Graph Learning for Real-Time Dynamic Job Shop Scheduling

This repository contains the environment, heterogeneous GNN policy, teacher
schedules, training scripts, and corrected JSPLIB evaluation scripts for the
L2D teacher-distillation paper.

## Status

- The manuscript is under revision; it is **not yet submission-ready**.
- The method is a two-stage teacher-distilled HGNN:
  1. CP-SAT optimal trajectories on small instances initialize a base policy.
  2. Official open-source L2D schedules on ta01-ta30 fine-tune the student.
- JSPLIB parsing preserves the published operation order.

## Results

- 49 official JSPLIB instances: HGNN beats SPT on 49/49, mean +25.2%.
- HGNN beats the best-of-five rule oracle on 33/49, mean difference -2.7%.
- Mean gap to known best is 34.0%, down from 56.9% for the released base
  policy.
- On 37 official instances with matching released L2D checkpoints, the student is on average 6.9% behind published L2D under `(student - teacher) / teacher`.
- A same-protocol Graph Transformer baseline is effectively tied with HGNN on the official suite (mean gap 34.0% vs 34.0%).
- CPU deployment: 2.4-3.2 ms per decision, 310-413 decisions/sec, 42,178 parameters.
- Training-protocol ablation on 49 official instances: CP-SAT-only gap 56.9%, L2D-only gap 36.7%, two-stage gap 34.0%, homogeneous two-stage gap 35.7%.

## Key files

| File | Purpose |
|---|---|
| `hetero_model_l2d.pt` | Distilled reference checkpoint |
| `hetero_train_data_l2d.pt` | 9,250 distillation states from ta01-ta30 |
| `L2D_official/collect_l2d_teacher.py` | Collect official L2D teacher schedules |
| `L2D_official/run_l2d_official_benchmarks.py` | Run published L2D on all matching official sizes |
| `L2D_official/results_l2d_official_match.json` | Published L2D results on 37 matching instances |
| `build_l2d_train_data.py` | Convert teacher schedules to HGNN states |
| `train_hetero_available.py` | Train HGNN with true available actions |
| `train_l2d_distill.py` | Train homogeneous baseline on the same teacher states |
| `hetero_model_l2d_only.pt` | L2D-only HGNN ablation checkpoint |
| `l2d_homogeneous_l2d.pt` | Homogeneous two-stage ablation checkpoint |
| `results_official_l2d_only.json` | L2D-only official results |
| `results_l2d_homogeneous_l2d.json` | Homogeneous two-stage official results |
| `run_corrected_official.py` | Official JSPLIB evaluation |
| `results_official_l2d_train.json` | Distilled model official results |
| `transformer_policy_l2d.pt` | Same-protocol Graph Transformer baseline |
| `results_transformer_l2d.json` | Graph Transformer official results |
| `deployment_cpu_latency.json` | CPU latency/throughput benchmark |
| `build_submit_paper_l2d.py` | Builds the current DOCX manuscript |

## Reproduce

```powershell
cd C:\Users\17302\Desktop\JSP_RL_Project
python train_hetero_available.py --data hetero_train_data_l2d.pt --out hetero_model_l2d.pt
python run_corrected_official.py hetero_model_l2d.pt results_official_l2d_train.json
```

The teacher schedules can be regenerated from `L2D_official` with the released
15x15, 20x15, and 20x20 checkpoints.

Large training-state files (`hetero_train_data_l2d.pt`, `hetero_train_data_large.pt`)
are not committed to GitHub because of file-size limits. They are available from
the corresponding author on request.
