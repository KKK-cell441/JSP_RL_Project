# Standard FJSP Benchmark Attempt Log

Date: 2026-08-07

## What I Did

1. Added standard FJSP support:
   - `fjsp_env.py`: FJSP environment with explicit machine alternatives.
   - `fjsp_benchmark_utils.py`: `.fjs` parser and SPT baseline.
   - `fjsp_solver.py`: corrected CP-SAT FJSP solver.
   - `run_fjsp_benchmarks.py`: standard benchmark evaluation.
2. Evaluated the existing `fjsp_policy.pt` on Brandimarte Mk01-Mk15 and
   Kacem1-Kacem2.
3. Collected 650 CP-SAT expert states from 8 representative standard instances
   and fine-tuned `fjsp_policy_standard.pt` with job + option loss.

## Results Summary

Existing policy on Brandimarte:

- Improves over SPT by roughly 26%-58%.
- Remains on average about 20%-30% above the 5s CP-SAT bound.
- Several instances are close to CP-SAT, e.g. Mk14 gap 4.2%, Mk09 gap 7.9%.

Fine-tuned policy on Brandimarte:

- Improves several instances, e.g. Mk13 gap 2.5%, Mk14 beats the 5s CP-SAT
  bound, Mk11 gap 11.1%.
- But the improvement is not consistent across all instances.

Kacem:

- Kacem1: existing policy 17 vs SPT 22 vs CP-SAT 11.
- Kacem2: existing policy 117 vs SPT 27 vs CP-SAT 11.
- Fine-tuned policy becomes worse on Kacem1/Kacem2 (67/66) because training is
  dominated by Brandimarte instances.
- The current job-level HGNN cannot handle high-flexibility Kacem instances
  with five machine alternatives per operation.

## Conclusion

The current FJSP extension is not strong enough to be reported as completed
standard benchmark validation. Adding these numbers to the JOIM paper would
create a new weakness rather than fixing one. A real FJSP extension requires an
operation-option level action space, not just fine-tuning the existing JSP
job-dispatch policy.

Recommendation: do not include FJSP as a completed contribution yet. Either
build a proper FJSP model or keep FJSP as future work and prioritize other
JOIM improvements (deployment scenario, more recent SOTA comparisons).
