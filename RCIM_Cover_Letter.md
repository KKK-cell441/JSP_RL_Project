Dear Editor-in-Chief,

We are pleased to submit our manuscript entitled "Heterogeneous Graph-based Learning-to-Dispatch for Dynamic Job Shop Scheduling with Topology Generalization" for consideration in Robotics and Computer-Integrated Manufacturing.

**Background and motivation.** Job shop scheduling (JSS) is a core decision problem in flexible manufacturing systems, where dispatching rules remain the industrial standard because they are fast and interpretable. However, no single rule performs consistently across shop configurations, dynamic arrivals, and production-line reconfigurations. Learning-based dispatch has shown promise, but most methods require retraining when the shop topology changes and rarely generalize to dynamic or failure-prone environments.

**What we propose.** We develop a Heterogeneous Graph Neural Network (HGNN) policy that explicitly encodes both job nodes and machine nodes, trained on CP-SAT optimal trajectories. The policy learns state-dependent operation selection that dynamically combines dispatching principles. A padding-and-masking mechanism enables zero-shot transfer across shop topologies without retraining. We evaluate the policy across static instances, Taillard-style large benchmarks, dynamic job arrivals, machine failures, and flexible job shop scenarios.

**Key findings.** Across 50 held-out 6x5 instances, the policy improves makespan by 9.1% over SPT (p < 0.0001, 41/50 wins). On Taillard-style 10x10, 15x15, and 20x15 instances, improvements are 16.0%, 16.2%, and 15.3% respectively (all p < 0.0001, 29-30/30 wins). The policy transfers zero-shot to unseen topologies, and outperforms SPT under dynamic arrivals (p = 0.021) and machine failures (p = 0.033). An ablation on 60 instances confirms that the heterogeneous representation adds a statistically significant 3.3% over a homogeneous baseline (p = 0.0016). The approach requires no hand-crafted features and executes in milliseconds, positioning it as a deployable alternative for real-time manufacturing scheduling.

**Why RCIM.** The manuscript addresses a core robotics and computer-integrated manufacturing concern: real-time, adaptive scheduling for reconfigurable production systems. The zero-shot topology generalization directly supports the reconfigurable manufacturing paradigm, and the dynamic and failure robustness aligns with resilient production requirements. We believe the contributions are well matched to the journal's scope.

This manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the manuscript and agree with its submission to Robotics and Computer-Integrated Manufacturing. The authors declare no conflict of interest.

Thank you for your consideration.

Sincerely,
[Author names and affiliations]

Corresponding author: [Name], [Email], [Address]
> SUPERSEDED: this RCIM cover letter contains outdated claims and should not be submitted. Use CAIE_Cover_Letter.md.
