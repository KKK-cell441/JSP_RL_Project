# Corrected Experimental Summary (CAIE Revision)

This file is the canonical summary after fixing the JSPLIB parser. The previous
`HETERO_RESULTS.md` values for the homogeneous-vs-heterogeneous ablation
(2.1%, p = 0.144, 18/30) and FJSP (+1.7% on 50 instances) are obsolete or were
not reproducible and are not used in the manuscript.

## Generated benchmark results

| Instance size | Hetero GNN | SPT | Improvement | t | p | Wins |
|---|---|---|---|---|---|---|
| 10x10 | 1291.3 +/- 83.9 | 1537.1 +/- 93.7 | +16.0% | -12.22 | <0.0001 | 29/30 |
| 15x15 | 2217.6 +/- 108.6 | 2646.8 +/- 146.9 | +16.2% | -16.43 | <0.0001 | 30/30 |
| 20x15 | 2723.8 +/- 142.0 | 3214.4 +/- 150.9 | +15.3% | -14.10 | <0.0001 | 30/30 |

These are generated instances with a fixed machine order, not official JSPLIB
instances. The manuscript labels them accordingly.

## Canonical heterogeneous-vs-homogeneous ablation

| Model | Mean +/- std | vs SPT | Hetero gain | t | p | Wins |
|---|---|---|---|---|---|---|
| Homogeneous L2D-style baseline | 1328.1 +/- 94.0 | +13.6% | - | - | - | - |
| Heterogeneous GNN | 1284.8 +/- 99.3 | +16.4% | +3.3% | -3.31 | 0.0016 | 41/60 |

Source: `results_homo_vs_hetero_60.json`.

## Official JSPLIB results (corrected parser)

- 49 instances from Taillard, Fisher-Thompson, Lawrence, Adams-Balazs-Zawack,
  ORB, and Storer-Wu-Vaccari.
- HGNN beats SPT on 44 of 49 instances; mean improvement vs SPT is 12.6%.
- HGNN loses to SPT on `ta11`, `abz6`, `ft20`, `swv01`, and `swv05`.
- Mean HGNN gap to known best is 56.9%.
- HGNN beats the best-of-five rule oracle on only 7 of 49 instances; mean
  HGNN-vs-oracle difference is -14.2% (HGNN worse).
- The best-of-five oracle's mean gap to known best is 38.0%.
- A same-protocol homogeneous L2D-CP baseline beats SPT on 25 of 49
  instances (mean +0.4%); HGNN beats L2D-CP on 39 of 49 (mean +11.1%).
- Official open-source L2D (Zhang et al., 2020) beats SPT on all 30 Taillard
  instances with pretrained checkpoints (mean +33.9%, mean known-best gap
  27.6%); HGNN is on average 28.1% worse under `(HGNN - L2D) / L2D`.
- The parser was validated by recovering the known optimum of `ta01` (1231).

Full per-instance table: `results_official_corrected.json`.
Baseline per-instance table: `results_l2d_cpsat_official.json`.
Official L2D per-instance table: `results_l2d_official_zhang.json`.

Reproduce with:

```powershell
cd C:\Users\17302\Desktop\JSP_RL_Project
python run_corrected_official.py
python run_corrected_warmstart.py
python run_l2d_baseline_official.py l2d_cpsat.pt results_l2d_cpsat_official.json
python train_hetero.py
python train_l2d_cpsat.py

cd C:\Users\17302\Desktop\JSP_RL_Project\L2D_official
python test_learned_on_benchmark.py --Pn_j 15 --Pn_m 15 --Nn_j 15 --Nn_m 15 --which_benchmark tai
python test_learned_on_benchmark.py --Pn_j 20 --Pn_m 15 --Nn_j 20 --Nn_m 15 --which_benchmark tai
python test_learned_on_benchmark.py --Pn_j 20 --Pn_m 20 --Nn_j 20 --Nn_m 20 --which_benchmark tai
```

## Warm start

On 8 official Taillard instances, CP-SAT with HGNN start-time hints improved
the 5-second feasible bound on 3 of 8 instances with a fixed solver seed, with
a maximum improvement of 0.62%. This is reported as a preliminary observation,
not a claimed speedup.

## Removed claims

- FJSP quantitative results are removed because the earlier experiment used
  unverifiable `opt_mk` values and no standard benchmark files.
- Any claim that HGNN outperforms SPT on all official instances is removed.
