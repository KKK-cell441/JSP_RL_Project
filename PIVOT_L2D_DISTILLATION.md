# Pivot: L2D Teacher Distillation

Date: 2026-08-07

## Why Pivot

Direct CP-SAT supervision on larger generated instances made the official
benchmark results worse instead of better. The architecture cannot absorb
enough signal from feasible CP-SAT schedules at 15x15/20x15/20x20 scale. I
therefore switched the supervision source from CP-SAT-only to official L2D
teacher schedules.

## What I Did

1. Collected official L2D schedules on ta01-ta30:
   - `L2D_official/l2d_teacher_15x15.json`
   - `L2D_official/l2d_teacher_20x15.json`
   - `L2D_official/l2d_teacher_20x20.json`
2. Converted them to HGNN training states:
   - `hetero_train_data_l2d.pt` (9,250 states)
3. Fine-tuned `hetero_model.pt` on those states:
   - `hetero_model_l2d.pt`
4. Evaluated greedy inference on 49 official JSPLIB instances:
   - `results_official_l2d_train.json`

## Results

Paper 34-instance subset:

| Metric | Original released model | L2D-distilled HGNN |
|---|---|---|
| Wins vs SPT | 29/34 | 34/34 |
| Wins vs best-of-five oracle | 7/34 | 25/34 |
| Mean difference vs oracle | +14.2% (worse) | -3.5% (better) |
| Mean gap to known best | 52.2% | 32.1% |

Full 49-instance suite:

| Metric | Original released model | L2D-distilled HGNN |
|---|---|---|
| Wins vs SPT | 44/49 | 49/49 |
| Wins vs best-of-five oracle | 7/49 | 33/49 |
| Mean difference vs oracle | +14.2% (worse) | -2.7% (better) |
| Mean gap to known best | 56.9% | 34.0% |

On the 30 official Taillard instances, the distilled student is still worse
than its teacher: it wins only 4/30 and is on average 8.4% behind official L2D.

## Current Status

The direction change is effective: the same lightweight HGNN now beats the
best-of-five rule oracle and cuts the known-best gap from roughly 52% to 32%.
It is not yet competitive with the official L2D teacher. This can be published
as a teacher-distillation / lightweight dispatch paper only if the framing is
changed accordingly. It is not yet a CAIE submission-ready method paper.
