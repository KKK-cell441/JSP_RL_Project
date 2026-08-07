#!/usr/bin/env python3
"""Evaluate the FJSP policy on standard Brandimarte and Kacem instances."""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fjsp_benchmark_utils import parse_fjs, build_env, run_spt
from fjsp_policy import FJSPHeteroPolicy, run_fjsp_policy
from fjsp_solver import solve_fjsp

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_ROOT = os.path.join(HERE, "FJSP_benchmarks_official")


def collect_files():
    files = []
    for sub in ["1_Brandimarte", "5_Kacem"]:
        root = os.path.join(BENCH_ROOT, sub)
        for name in sorted(os.listdir(root)):
            if name.lower().endswith(".fjs"):
                files.append((name, os.path.join(root, name), sub))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fjsp_policy.pt")
    parser.add_argument("--out", default="results_fjsp_standard.json")
    parser.add_argument("--time-limit", type=float, default=5.0)
    args = parser.parse_args()

    agent = FJSPHeteroPolicy(hidden_dim=64, n_layers=2, n_max_jobs=30).eval()
    agent.load_state_dict(torch.load(os.path.join(HERE, args.model), map_location="cpu"))

    results = {}
    for name, path, family in collect_files():
        n_jobs, n_mach, alt = parse_fjs(path)
        env = build_env(n_jobs, n_mach, alt)
        t0 = time.time()
        agent_mk, agent_ft = run_fjsp_policy(agent, env)
        agent_t = time.time() - t0
        spt_mk, spt_ft = run_spt(env)
        t1 = time.time()
        cp_mk, cp_ft, _ = solve_fjsp(alt, time_limit=args.time_limit)
        cp_t = time.time() - t1
        results[name] = {
            "family": family,
            "size": f"{n_jobs}x{n_mach}",
            "n_ops": sum(len(ops) for ops in alt),
            "agent_mk": float(agent_mk),
            "agent_ft": float(agent_ft),
            "agent_seconds": agent_t,
            "spt_mk": float(spt_mk),
            "spt_ft": float(spt_ft),
            "cp_mk": cp_mk,
            "cp_ft": cp_ft,
            "cp_seconds": cp_t,
            "improve_vs_spt": (spt_mk - agent_mk) / spt_mk * 100 if spt_mk else None,
            "gap_vs_cp": (agent_mk - cp_mk) / cp_mk * 100 if cp_mk else None,
        }
        print(
            f"{name}: {n_jobs}x{n_mach} HGNN={agent_mk:.0f} SPT={spt_mk:.0f} "
            f"CP={cp_mk} impr={results[name]['improve_vs_spt']:.1f}% "
            f"gap={results[name]['gap_vs_cp']:.1f}%",
            flush=True,
        )

    with open(os.path.join(HERE, args.out), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
