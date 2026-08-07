# FJSP Benchmark Template and Data Availability Paragraph

## Data Availability paragraph (paste into manuscript)

All source code, benchmark instances, raw experimental results, and training
scripts are available at the public GitHub repository: [https://github.com/xxx].
A permanent Zenodo DOI is provided at release: [https://zenodo.org/xxx].
All experiments can be fully reproduced with the fixed random seeds provided in
the repository. The official open-source L2D implementation used for baseline
comparison is available at https://github.com/XUZiteng2020/L2D.

## BRdata + Kacem result table template

| Instance | Size (n x m) | Flexibility | HGNN MK | SPT MK | Official L2D MK | GA MK | BKS / CP-SAT | HGNN gap to BKS |
|---|---|---|---|---|---|---|---|---|
| Kacem1 | 4 x 5 | ... | ... | ... | ... | ... | ... | ... |
| Kacem2 | 10 x 7 | ... | ... | ... | ... | ... | ... | ... |
| Kacem3 | 10 x 10 | ... | ... | ... | ... | ... | ... | ... |
| Kacem4 | 15 x 10 | ... | ... | ... | ... | ... | ... | ... |
| Mk01 | 10 x 6 | ... | ... | ... | ... | ... | ... | ... |
| Mk02 | 10 x 6 | ... | ... | ... | ... | ... | ... | ... |
| Mk03 | 15 x 8 | ... | ... | ... | ... | ... | ... | ... |
| Mk04 | 15 x 8 | ... | ... | ... | ... | ... | ... | ... |
| Mk05 | 15 x 8 | ... | ... | ... | ... | ... | ... | ... |
| Mk06 | 10 x 15 | ... | ... | ... | ... | ... | ... | ... |
| Mk07 | 20 x 5 | ... | ... | ... | ... | ... | ... | ... |
| Mk08 | 20 x 10 | ... | ... | ... | ... | ... | ... | ... |
| Mk09 | 20 x 10 | ... | ... | ... | ... | ... | ... | ... |
| Mk10 | 20 x 15 | ... | ... | ... | ... | ... | ... | ... |

## Status

- Benchmark files are cloned in `FJSP_benchmarks_official` from
  https://github.com/leikun-starting/FJSP-benchmarks.
- No FJSP results are reported in the current manuscript yet.
- The current HGNN policy does not yet have a reproducible FJSP routing
  extension, so results must not be filled in until the model and environment
  support standard FJSP action spaces.
