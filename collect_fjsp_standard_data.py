#!/usr/bin/env python3
"""Collect FJSP expert states from standard benchmarks using CP-SAT schedules."""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fjsp_benchmark_utils import parse_fjs, build_env
from fjsp_solver import solve_fjsp
from fjsp_policy import FJSPHeteroPolicy


def collect_from_alt(alt, agent, time_limit=5.0):
    n_jobs = len(alt)
    n_machines = max(m for job in alt for opts in job for m, _ in opts) + 1
    mk, ft, schedule = solve_fjsp(alt, time_limit=time_limit)
    if mk is None:
        return []
    env = build_env(n_jobs, n_machines, alt)
    env.reset()
    entries = []
    while not env.done:
        avail = env._get_available_operations()
        if not avail:
            env._advance_time()
            continue
        best_j = min(avail, key=lambda j: schedule[(j, env.job_progress[j])][1])
        k = env.job_progress[best_j]
        m = schedule[(best_j, k)][0]
        oi = next(i for i, (mm, _) in enumerate(alt[best_j][k]) if mm == m)
        jf = agent.get_job_features(env)
        mf = agent.get_machine_features(env)
        je, me, jme = agent.build_edges(env)
        opts = alt[best_j][k]
        opt_feats = torch.FloatTensor([
            [mm / max(n_machines, 1), dd / (dd + 1e-8)] for mm, dd in opts
        ])
        entries.append((
            jf.clone(), mf.clone(), je.clone(), me.clone(), jme.clone(),
            int(best_j), int(oi), opt_feats.clone(), list(avail),
        ))
        env.step_option(best_j, oi)
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=3.0)
    parser.add_argument("--max-files", type=int, default=8)
    args = parser.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    agent = FJSPHeteroPolicy(hidden_dim=64, n_layers=2, n_max_jobs=30).eval()
    selected = [
        ("5_Kacem", "Kacem1.fjs"),
        ("5_Kacem", "Kacem2.fjs"),
        ("1_Brandimarte", "BrandimarteMk1.fjs"),
        ("1_Brandimarte", "BrandimarteMk2.fjs"),
        ("1_Brandimarte", "BrandimarteMk3.fjs"),
        ("1_Brandimarte", "BrandimarteMk4.fjs"),
        ("1_Brandimarte", "BrandimarteMk5.fjs"),
        ("1_Brandimarte", "BrandimarteMk6.fjs"),
    ][:args.max_files]
    files = [os.path.join(here, "FJSP_benchmarks_official", sub, name) for sub, name in selected]
    entries = []
    t0 = time.time()
    for path in files:
        n_jobs, n_mach, alt = parse_fjs(path)
        ep = collect_from_alt(alt, agent, time_limit=args.time_limit)
        entries.extend(ep)
        print(f"{os.path.basename(path)}: states={len(ep)} elapsed={time.time()-t0:.1f}s", flush=True)
    out = os.path.join(here, "fjsp_train_data_standard.pt")
    torch.save(entries, out)
    print(f"Saved {len(entries)} states to {out}")


if __name__ == "__main__":
    main()
