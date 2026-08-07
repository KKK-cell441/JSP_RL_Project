# Method Improvement Experiment Log

Date: 2026-08-07

## Goal

Try to close the gap between the current HGNN and published L2D / best-of-five
rule oracle by training on larger instances or adding stronger inference.

## What I Tried

### 1. Larger CP-SAT expert training set

Created `collect_large_expert_data.py` and collected 4,750 states:

- 10 generated 10x10 instances
- 10 generated 15x15 instances
- 5 generated 20x15 instances
- CP-SAT time limit 3s per instance

Trained `hetero_model_large.pt` from the released checkpoint with
`train_hetero_available.py` for 5 epochs.

Result on the paper's 34 official instances:

| Model | Wins vs SPT | Mean vs SPT | Wins vs oracle | Mean gap to BKS |
|---|---|---|---|---|
| Released model | 29/34 | +11.9% | 7/34 | 52.2% |
| Large-trained model | 16/34 | -2.0% | 5/34 | 77.8% |

Conclusion: simply adding more CP-SAT expert states from the same architecture
made official Taillard performance worse. Loss decreased only from 0.527 to
0.517, and the current job-level state representation does not transfer the
CP-SAT advantage to official routes.

### 2. Policy-guided beam search

Added `beam_search_eval.py`.

On 49 official instances with beam width 4:

| Metric | Released greedy | Beam width 4 |
|---|---|---|
| Wins vs SPT | 44/49 | 48/49 |
| Mean improvement vs SPT | +12.6% | +24.2% |
| Wins vs best-of-five oracle | 7/49 | 32/49 |
| Mean difference vs oracle | +14.2% (HGNN worse) | -1.2% (beam slightly better) |
| Mean gap to known best | 56.9% | 35.5% |
| Total inference time | n/a | 354s for 49 instances |

On 30 official Taillard instances, the released official L2D baseline is still
better: beam width 4 wins only 4/30 and is on average 7.7% worse than L2D.

Beam width 8 and a hybrid policy+rule beam did not improve over width 4.

## Main Difficulties

1. CUDA initialization hung, so experiments ran on CPU. Large-instance training
   and 49-instance beam evaluation are slow.
2. CP-SAT labels on 15x15/20x15 are only 3-5 second feasible bounds, not
   proven optima, and the current architecture cannot translate them into a
   better zero-shot dispatcher.
3. The job-level feature set is too coarse: current operation duration,
   remaining work, total work, progress, and machine load. It does not model
   full operation sequences or disjunctive graph structure, which is why it
   cannot compete with L2D on hard Taillard instances.
4. Beam search improves greedy dispatch substantially, but it changes the
   method from millisecond online dispatch to seconds-per-instance search and
   still does not consistently beat published L2D.

## Next Steps That Would Actually Move the Needle

- Replace/augment the job-level encoder with an operation-level or
  disjunctive-graph state representation, similar to L2D-style features.
- Train on official-size instances with better labels, then fine-tune with RL
  or expert iteration using beam-search-improved schedules.
- If search is acceptable, add local search or beam search as an explicit
  hybrid method and compare fairly with L2D at similar inference budgets.
- Extend the official L2D reproduction to the full 49-instance suite or train
  a same-protocol RL/L2D baseline on every non-covered size.
