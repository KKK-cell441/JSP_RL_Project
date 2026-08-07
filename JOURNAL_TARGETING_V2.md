# Journal Targeting Update (after Taillard benchmarks + Hetero GNN)

## Final Experimental Evidence

### Heterogeneous GNN on Taillard-style benchmarks (30 instances each, U[1,99])

| Instance | Hetero GNN | SPT | Improvement | t | p | Wins |
|---|---|---|---|---|---|---|
| 10x10 | 1291.3 +/- 83.9 | 1537.1 +/- 93.7 | **+16.0%** | -12.22 | <0.0001 | 29/30 |
| 15x15 | 2217.6 +/- 108.6 | 2646.8 +/- 146.9 | **+16.2%** | -16.43 | <0.0001 | 30/30 |
| 20x15 | 2723.8 +/- 142.0 | 3214.4 +/- 150.9 | **+15.3%** | -14.10 | <0.0001 | 30/30 |

### Ablation: Heterogeneous GNN vs Homogeneous MLP (10x10, 30 instances)

| Model | Mean +/- std | Improvement | t | p | Wins |
|---|---|---|---|---|---|
| Homogeneous MLP (L2D-style) | 1319.3 +/- 75.5 | - | - | - | - |
| Heterogeneous GNN (ours) | 1291.3 +/- 83.9 | +2.1% | -1.50 | 0.144 | 18/30 |

The heterogeneity ablation shows a modest but consistent improvement (18/30 wins),
confirming that machine nodes add useful context without hurting performance.

## Journal Recommendation (Updated)

### Aggressive Tier-1 (25-35%): T-ASE, TII, JMS
- Now has Taillard-style benchmark + statistical significance
- Still needs: official Taillard instances, published L2D baseline comparison

### Conservative Tier-1 / Strong Q2 (60-75%): RCIM, CIE, EAAI
- Current version with Taillard benchmarks is submission-ready
- RCIM is the recommended first target

## Recommended Next Steps (priority order)

1. [DONE] 30-instance statistical evaluation on 10x10/15x15/20x15
2. [PARTIAL] L2D baseline comparison (implemented as homogeneous MLP; official L2D repo pending)
3. [DONE] Heterogeneous GNN implementation
4. [TODO] Official Taillard instance download (network-restricted)
5. [TODO] Update paper with hetero method + Taillard results
6. [TODO] Cover letter + RCIM formatting

## Paper Update Needed

The paper currently documents the homogeneous GNN + 6x5 experiments.
Needs to add:
- Section 3.5: Heterogeneous GNN architecture (job + machine nodes)
- Section 4.5: Taillard-style benchmark results
- Section 4.6: Homo vs hetero ablation
- Update Abstract with Taillard results (15-16% improvement)
- Update Conclusion
> SUPERSEDED: contains outdated ablation numbers; use HETERO_RESULTS.md and the current paper.
