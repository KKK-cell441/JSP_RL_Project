# JSP RL Paper - Journal Targeting Assessment

## Current Paper Status (after hetero GNN + Taillard-style benchmarks)

### Experimental Evidence

| Experiment | Agent | SPT | Improvement | Significance |
|---|---|---|---|---|
| Static 6x5 (50 instances) | 129.4 | 142.4 | +9.1% | p < 0.0001 |
| Topology 5x5 (zero-shot) | 122.5 | 123.8 | +1.0% | - |
| Topology 8x5 (zero-shot) | 164.4 | 184.1 | +10.7% | - |
| Topology 6x8 (zero-shot) | 179.5 | 201.4 | +10.9% | - |
| Dynamic arrivals (50) | 175.3 | 192.0 | +8.7% | p = 0.021 |
| Machine failure (50) | 182.4 | 201.7 | +9.6% | p = 0.033 |
| FJSP zero-shot (50) | 136.7 | 139.1 | +1.7% | n.s. |
| Taillard-style 10x10 (6) | 1316.8 | 1564.5 | **+15.8%** | - |
| Taillard-style 15x15 (6) | 2303.2 | 2652.5 | **+13.2%** | - |
| Taillard-style 20x15 (4) | 2677.5 | 3178.5 | **+15.8%** | - |

### Method Assets
- CP-SAT optimal-supervised GNN policy
- Zero-shot topology generalization via padding/masking
- Heterogeneous GNN (job + machine nodes)
- Dynamic arrival + machine failure robustness
- FJSP environment extension

## Journal Targeting

### Aggressive Tier-1 (worth trying, 25-35% chance)

1. **IEEE Transactions on Automation Science and Engineering (T-ASE)**
   - IF ~5.5, CCF B, 中科院二区 (但 IEEE 自动化顶刊)
   - 匹配度: RL scheduling 是 T-ASE 常客
   - 风险: 需要更严格 benchmark + 与 SOTA 对比

2. **Journal of Manufacturing Systems (JMS)**
   - IF ~12, 中科院一区
   - 匹配度: manufacturing + ML 完美匹配
   - 风险: 一区竞争激烈，需要补 SOTA baseline 对比

3. **IEEE Transactions on Industrial Informatics (TII)**
   - IF ~11, 中科院一区
   - 匹配度: 工业调度 + GNN 强相关
   - 风险: 需要工业场景验证或更大规模实验

### Conservative Tier-1 / Strong Q2 (60-75% chance)

1. **Robotics and Computer-Integrated Manufacturing (RCIM)**
   - IF ~10, 中科院一区（新晋）
   - 匹配度: 智能制造调度高频主题
   - 当前完整度: 接近可投

2. **Computers & Industrial Engineering (CIE)**
   - IF ~7, 中科院二区 Top
   - 匹配度: 工业工程 + 调度经典期刊
   - 当前完整度: 已足够，可投

3. **Engineering Applications of Artificial Intelligence (EAAI)**
   - IF ~8, 中科院一区（波动）
   - 匹配度: AI 应用 + 调度
   - 当前完整度: 需补 benchmark 对比

## Recommended Strategy

### Phase 1 (now): Strengthen for T-ASE / JMS
1. 补标准 Taillard benchmark 官方实例（当前为 style 生成器）
2. 实现 L2D baseline 对比（Zhang et al. 2020）
3. 补消融实验（hetero vs homo GNN）
4. 补 30 个实例的统计显著性（当前 6 个偏少）

### Phase 2: Submit RCIM first (safest Q1)
- 当前版本 + 补齐统计即可
- 审稿周期 2-3 个月
- 录用概率 60-70%

### Phase 3: If rejected, submit JMS / T-ASE
- 带审稿意见修改
- 补 SOTA 对比后冲一区

## What Still Needs Work Before Submission

| Item | Effort | Priority |
|---|---|---|
| Official Taillard benchmark download | Low (需外网) | High |
| L2D baseline implementation | Medium | High |
| 30-instance statistical evaluation | Low | High |
| Hetero vs homo ablation | Medium | Medium |
| Cover letter | Low | Medium |
| Journal format (RCIM Elsevier template) | Medium | Medium |
> SUPERSEDED: contains outdated journal-targeting claims and should not be used for the current CAIE revision.
