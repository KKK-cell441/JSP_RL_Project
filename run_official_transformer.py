#!/usr/bin/env python3
"""Evaluate the Graph Transformer policy on the paper's 49 official JSPLIB instances."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jsp_env import JSPEnv, schedule_with_heuristic
from transformer_policy import TransformerPolicy
from run_corrected_official import parse_jsplib, collect_instances, best_known

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "transformer_policy_l2d.pt"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "results_transformer_l2d.json"

    agent = TransformerPolicy(hidden_dim=64, n_heads=4, n_layers=2, n_max_jobs=30).eval()
    agent.load_state_dict(torch.load(os.path.join(HERE, model_path), map_location="cpu"))

    reference = json.load(open(os.path.join(HERE, "results_official_l2d_train.json"), encoding="utf-8"))
    best_data = {rec["name"]: rec for rec in json.load(open(os.path.join(HERE, "jsplib_instances.json"), encoding="utf-8"))}
    results = {}
    for name, path, family in collect_instances():
        if name not in reference:
            continue
        n_jobs, n_mach, ops = parse_jsplib(path)
        env = JSPEnv.from_operations(n_jobs, n_mach, ops)
        env.reset()
        agent.eval()
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
        mk = env.get_makespan()
        spt = schedule_with_heuristic(env, "SPT")
        best_rule = min(schedule_with_heuristic(env, r) for r in ["FIFO", "SPT", "LPT", "MWKR", "LWKR"])
        bks, _ = best_known(name, best_data)
        results[name] = {
            "size": f"{n_jobs}x{n_mach}",
            "transformer_mk": float(mk),
            "spt": float(spt),
            "best_rule": float(best_rule),
            "known_best": bks,
            "impr_vs_spt": (spt - mk) / spt * 100 if spt else None,
            "gap_bks": (mk - bks) / bks * 100 if bks else None,
        }
        print(f"{name}: transformer={mk:.0f} spt={spt:.0f} bks={bks} gap={results[name]['gap_bks']:.1f}%", flush=True)

    with open(os.path.join(HERE, out_path), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
