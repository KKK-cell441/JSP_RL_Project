# JOIM Submission Package (Teacher-Distilled HGNN Draft)

## Highlights

1. Teacher-distilled HGNN dispatcher for job shop scheduling
2. Variable-size heterogeneous graph supports topology transfer
3. Beats SPT on 49 of 49 official JSPLIB instances
4. Beats best-of-five oracle on 33 of 49 instances
5. Robust to dynamic arrivals and machine failures in simulation

## Title Page

**Title:** Teacher-Distilled Lightweight Graph Learning for Real-Time Dynamic Job Shop Scheduling

**Authors:** ZHONGKUAN MA

**Affiliation:** Northeast Forestry University

**Corresponding author:** ZHONGKUAN MA, 2024212760@nefu.edu.cn, Northeast Forestry University

## Manuscript Status

| Item | Status | Notes |
|---|---|---|
| Corrected JSPLIB parser | Done | Preserves operation order; ta01 optimum 1231 reproduced |
| Official 49-instance table | Done | `results_official_l2d_train.json` |
| L2D teacher schedules | Done | ta01-ta30 with released 15x15/20x15/20x20 models |
| Teacher distillation training | Done | `hetero_model_l2d.pt`, `hetero_train_data_l2d.pt` |
| Training-supervision ablation | Done | CP-SAT-only, L2D-only, two-stage, homogeneous two-stage |
| Published L2D comparison | Partial | Published L2D on 37 matching official sizes; 10x5/15x5/20x5/20x10 still missing |
| Same-protocol Graph Transformer | Done | `results_transformer_l2d.json` |
| CPU deployment benchmark | Done | `deployment_cpu_latency.json` |
| FJSP benchmark | Not done | BRdata/Kacem remain future work |
| Industrial/semi-real validation | Not done | Required for a strong JOIM submission |
| Author/title page | Partially done | Author, affiliation, and email added; address/repo still needed |
| Repository URL | Done | https://github.com/KKK-cell441/JSP_RL_Project |

## Submission Files Checklist (Springer JOIM)

| File | Status | Notes |
|---|---|---|
| Manuscript (Word/LaTeX) | `JSP_RL_Paper_L2D_Distill.docx` | Revised draft, not yet submission-ready |
| Cover Letter | `JOIM_Cover_Letter.md` | Updated to teacher-distillation framing |
| Highlights | In this file | 5 bullets |
| Title Page | In this file | Fill authors/affiliations |
| Figures | fig1-5.png in project | Verify 300 dpi |
| Tables | in manuscript | 16 numbered tables |
| Declaration of Interest | in manuscript | No conflict |
| Data Availability | in manuscript | Repository placeholder |

## Required Work Before JOIM Submission

1. Decide whether JOIM is the right venue for a teacher-student distillation
   paper; current results do not beat the published L2D teacher.
2. Add published L2D/RL comparisons on the remaining benchmark families, or
   clearly scope the paper as a lightweight distillation study.
3. Add standard FJSP benchmarks or keep FJSP as explicit future work.
4. Add a small semi-real or industrial validation scenario.
5. Fill authors, affiliations, corresponding author, and public repository URL.
6. Re-check every number against `results_*.json` and the released scripts.
