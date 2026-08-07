#!/usr/bin/env python3
"""Hybrid policy+rule beam search for JSS dispatch."""
import argparse
import copy
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jsp_env import JSPEnv
from hetero_model import HeteroPolicy
from run_corrected_official import parse_jsplib


def lower_bound(env):
    return max(max(env.machine_free), env.time)


def rule_action(env, rule):
    avail = env._get_available_operations()
    if not avail:
        return None
    ops = env.operations
    prog = env.job_progress
    if rule == "FIFO":
        return min(avail)
    if rule == "SPT":
        return min(avail, key=lambda j: ops[j][prog[j]].duration)
    if rule == "LPT":
        return max(avail, key=lambda j: ops[j][prog[j]].duration)
    if rule == "MWKR":
        return max(avail, key=lambda j: sum(o.duration for o in ops[j][prog[j]:]))
    if rule == "LWKR":
        return min(avail, key=lambda j: sum(o.duration for o in ops[j][prog[j]:]))
    return None


def run_hybrid_beam(agent, env, width, top_k, max_states=500000):
    env.reset()
    beams = [env]
    steps = 0
    rules = ["FIFO", "SPT", "LPT", "MWKR", "LWKR"]
    while beams and not all(b.done for b in beams) and steps < max_states:
        next_beams = []
        for e in beams:
            if e.done:
                next_beams.append(e)
                continue
            avail = e._get_available_operations()
            if not avail:
                e._advance_time()
                next_beams.append(e)
                continue
            candidates = set()
            for r in rules:
                a = rule_action(e, r)
                if a is not None:
                    candidates.add(a)
            logits, _, _ = agent.forward(e, available=avail)
            order = torch.argsort(logits, descending=True).tolist()
            for rank in order[:top_k]:
                candidates.add(avail[rank])
            candidates = list(candidates)[:width]
            for action in candidates:
                ne = copy.deepcopy(e)
                ne.step(action)
                next_beams.append(ne)
        next_beams.sort(key=lower_bound)
        beams = next_beams[:width]
        steps += 1
    return min(e.get_makespan() for e in beams), steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hetero_model.pt")
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", default="")
    parser.add_argument("--instances", nargs="*", default=None)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    agent = HeteroPolicy(hidden_dim=64, n_layers=2, n_max_jobs=30).eval()
    agent.load_state_dict(torch.load(os.path.join(here, args.model), map_location="cpu"))

    best_data = {rec["name"]: rec for rec in json.load(open(os.path.join(here, "jsplib_instances.json"), encoding="utf-8"))}
    reference = json.load(open(os.path.join(here, "results_official_corrected.json"), encoding="utf-8"))

    if args.instances is None:
        instances = list(reference.keys())
    else:
        instances = args.instances

    root = os.path.join(here, "taillard")
    bench_root = os.path.join(here, "benchmarks")
    results = {}
    for name in instances:
        path = os.path.join(root, name) if os.path.exists(os.path.join(root, name)) else os.path.join(bench_root, name)
        n_jobs, n_mach, ops = parse_jsplib(path)
        env = JSPEnv.from_operations(n_jobs, n_mach, ops)
        t0 = time.time()
        mk, steps = run_hybrid_beam(agent, env, args.width, args.top_k)
        bks = best_data.get(name, {}).get("optimum")
        if bks is None:
            bks = best_data.get(name, {}).get("bounds", {}).get("upper")
        ref = reference.get(name, {})
        gap = (mk - bks) / bks * 100 if bks else None
        results[name] = {
            "hybrid_beam_mk": float(mk),
            "greedy_mk": float(ref.get("hgmn", float("nan"))),
            "spt": float(ref.get("spt", float("nan"))),
            "best_rule": float(ref.get("best_rule", float("nan"))),
            "known_best": bks,
            "gap_bks_pct": gap,
            "steps": steps,
            "seconds": time.time() - t0,
        }
        print(f"{name}: beam={mk:.0f} bks={bks} gap={gap:.1f}% time={results[name]['seconds']:.1f}s", flush=True)

    if args.out:
        with open(os.path.join(here, args.out), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
