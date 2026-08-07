#!/usr/bin/env python3
"""Collect CP-SAT expert states for generated larger job shop instances.

The saved entries have the same graph tensors as the existing 6x5 data, plus
the true available action list. This makes the training target conditional on
the set of jobs that can actually be dispatched at each state.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jsp_env import JSPEnv
from hetero_model import HeteroPolicy


def solve_cp(ops, n_mach, time_limit=5.0):
    model = cp_model.CpModel()
    horizon = int(sum(d for job in ops for _, d in job)) * 2
    starts, ends = {}, {}
    machine_ops = [[] for _ in range(n_mach)]
    for j, job in enumerate(ops):
        prev = None
        for k, (m, dur) in enumerate(job):
            s = model.NewIntVar(0, horizon, f"s_{j}_{k}")
            e = model.NewIntVar(0, horizon, f"e_{j}_{k}")
            model.Add(e - s == int(dur))
            iv = model.NewIntervalVar(s, int(dur), e, f"i_{j}_{k}")
            starts[(j, k)] = s
            ends[(j, k)] = e
            machine_ops[m].append(iv)
            if prev is not None:
                model.Add(s >= prev)
            prev = e
    for mo in machine_ops:
        if mo:
            model.AddNoOverlap(mo)
    makespan = model.NewIntVar(0, horizon, "mk")
    for e in ends.values():
        model.Add(makespan >= e)
    model.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = {}
        for key in starts:
            schedule[key] = solver.Value(starts[key])
        return float(solver.ObjectiveValue()), schedule
    return None, None


def fixed_route_ops(pt):
    n_jobs, n_machines = pt.shape
    ops = []
    for j in range(n_jobs):
        seq = [(m, float(pt[j, m])) for m in range(n_machines)]
        ops.append(seq)
    return ops


def collect_episode(env, schedule, agent):
    env.reset()
    entries = []
    while not env.done:
        avail = env._get_available_operations()
        if not avail:
            env._advance_time()
            continue
        if schedule is None:
            break
        # The expert follows the CP-SAT schedule: dispatch the available job
        # whose next operation starts earliest in that solution.
        best = min(avail, key=lambda j: schedule[(j, env.job_progress[j])])
        job_feats = agent.get_job_features(env)
        machine_feats = agent.get_machine_features(env)
        je, me, jme = agent.build_edges(env)
        entries.append((
            job_feats.clone(),
            machine_feats.clone(),
            je.clone(),
            me.clone(),
            jme.clone(),
            int(best),
            list(avail),
        ))
        env.step(best)
    return entries


def collect_generated(spec, seed_start, n_instances, time_limit, agent):
    n_jobs, n_machines, max_dur = spec
    entries = []
    times = []
    for i in range(n_instances):
        seed = seed_start + i
        pt = JSPEnv.generate_random_instance(
            n_jobs, n_machines, min_dur=1, max_dur=max_dur, seed=seed
        )
        ops = fixed_route_ops(pt)
        env = JSPEnv.from_operations(n_jobs, n_machines, ops)
        t0 = time.time()
        mk, schedule = solve_cp(ops, n_machines, time_limit=time_limit)
        dt = time.time() - t0
        if mk is None:
            continue
        ep = collect_episode(env, schedule, agent)
        entries.extend(ep)
        times.append(dt)
        print(
            f"gen {n_jobs}x{n_machines} seed={seed} mk={mk:.0f} "
            f"states={len(ep)} cp_time={dt:.2f}s",
            flush=True,
        )
    return entries, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="hetero_train_data_large.pt")
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=30000)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, args.out)
    agent = HeteroPolicy(hidden_dim=64, n_layers=2, n_max_jobs=30).eval()

    specs = [
        (10, 10, 99, args.instances),
        (15, 15, 99, args.instances),
        (20, 15, 99, max(3, args.instances // 2)),
    ]
    all_entries = []
    all_times = []
    offset = 0
    for n_jobs, n_machines, max_dur, count in specs:
        entries, times = collect_generated(
            (n_jobs, n_machines, max_dur),
            args.seed_start + offset,
            count,
            args.time_limit,
            agent,
        )
        all_entries.extend(entries)
        all_times.extend(times)
        offset += count

    torch.save(all_entries, out_path)
    meta_path = out_path.replace(".pt", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "instances": args.instances,
                "time_limit": args.time_limit,
                "states": len(all_entries),
                "mean_cp_seconds": float(np.mean(all_times)) if all_times else None,
                "total_cp_seconds": float(np.sum(all_times)),
            },
            f,
            indent=2,
        )
    print(f"Saved {len(all_entries)} states to {out_path}")


if __name__ == "__main__":
    main()
