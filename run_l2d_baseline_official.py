#!/usr/bin/env python3
"""Evaluate the released L2D-style homogeneous baseline on official JSPLIB instances."""
import json
import os
import sys
import torch
from ppo_agent import PPOAgent
from jsp_env import JSPEnv, schedule_with_heuristic
from run_corrected_official import parse_jsplib, collect_instances, best_known

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\17302\Desktop\JSP_RL_Project"
WORK = HERE
BEST_JSON = os.path.join(WORK, "jsplib_instances.json")
MODEL = os.path.join(WORK, "l2d_final.pt") if len(sys.argv) < 2 else sys.argv[1]
OUT = os.path.join(WORK, "results_l2d_official.json") if len(sys.argv) < 3 else sys.argv[2]

device = "cuda" if torch.cuda.is_available() else "cpu"
agent = PPOAgent(hidden_dim=64, n_layers=2).to(device)
agent.load_state_dict(torch.load(MODEL, map_location=device))
agent.eval()


def run_l2d(env):
    env.reset()
    with torch.no_grad():
        while not env.done:
            avail = env._get_available_operations()
            if not avail:
                env._advance_time()
                continue
            job, _, _ = agent.get_action(env, deterministic=True)
            if job is None:
                break
            env.step(job)
    return env.get_makespan()


with open(BEST_JSON, encoding="utf-8") as f:
    best_data = {rec["name"]: rec for rec in json.load(f)}

results = {}
for name, path, family in collect_instances():
    n_jobs, n_mach, ops = parse_jsplib(path)
    env = JSPEnv.from_operations(n_jobs, n_mach, ops)
    agent_mk = run_l2d(env)
    spt = schedule_with_heuristic(env, "SPT")
    best_rule = min(schedule_with_heuristic(env, r) for r in ["FIFO", "SPT", "LPT", "MWKR", "LWKR"])
    bks, _ = best_known(name, best_data)
    results[name] = {
        "size": f"{n_jobs}x{n_mach}",
        "l2d": float(agent_mk),
        "spt": float(spt),
        "best_rule": float(best_rule),
        "known_best": bks,
        "l2d_vs_spt_pct": (spt - agent_mk) / spt * 100,
        "l2d_gap_bks_pct": ((agent_mk - bks) / bks * 100) if bks else None,
    }
    print(f"{name:8s} L2D={agent_mk:6.0f} SPT={spt:6.0f} "
          f"(+{(spt-agent_mk)/spt*100:5.1f}%) BKS={bks if bks else '-':<6}")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"Saved {OUT}")
