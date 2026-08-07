#!/usr/bin/env python3
"""CPU deployment benchmark for the distilled HGNN policy."""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jsp_env import JSPEnv
from hetero_model import HeteroPolicy


def run_agent(agent, env):
    env.reset()
    n_decisions = 0
    with torch.no_grad():
        while not env.done:
            avail = env._get_available_operations()
            if not avail:
                env._advance_time()
                continue
            action = agent.get_action(env, deterministic=True)[0]
            if action is None:
                break
            env.step(action)
            n_decisions += 1
    return env.get_makespan(), n_decisions


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    agent = HeteroPolicy(hidden_dim=64, n_layers=2, n_max_jobs=30).eval()
    agent.load_state_dict(torch.load(os.path.join(here, "hetero_model_l2d.pt"), map_location="cpu"))
    n_params = sum(p.numel() for p in agent.parameters())

    sizes = [(6, 5), (10, 10), (15, 15), (20, 20)]
    results = {}
    for nj, nm in sizes:
        total_time, total_decisions, total_mk = 0.0, 0, 0.0
        for seed in range(10):
            pt = JSPEnv.generate_random_instance(nj, nm, min_dur=1, max_dur=99, seed=100000 + seed)
            env = JSPEnv(nj, nm, pt)
            t0 = time.perf_counter()
            mk, nd = run_agent(agent, env)
            total_time += time.perf_counter() - t0
            total_decisions += nd
            total_mk += mk
        per_decision = total_time / total_decisions * 1000
        throughput = total_decisions / total_time
        results[f"{nj}x{nm}"] = {
            "instances": 10,
            "total_seconds": round(total_time, 3),
            "decisions": total_decisions,
            "per_decision_ms": round(per_decision, 3),
            "throughput_decisions_per_sec": round(throughput, 2),
            "mean_makespan": round(total_mk / 10, 2),
        }
        print(f"{nj}x{nm}: per_decision={per_decision:.3f}ms throughput={throughput:.1f}/s", flush=True)

    out = {
        "device": "cpu",
        "model": "hetero_model_l2d.pt",
        "parameters": n_params,
        "size_mb": round(n_params * 4 / 1024 / 1024, 3),
        "sizes": results,
    }
    with open(os.path.join(here, "deployment_cpu_latency.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved deployment_cpu_latency.json; params={n_params}")


if __name__ == "__main__":
    main()
