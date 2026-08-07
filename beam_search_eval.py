#!/usr/bin/env python3
"""Policy-guided beam search for JSS dispatch.

Keeps the best W partial schedules by the current makespan lower bound, using
the learned policy logits to rank candidate actions within each beam.
"""
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

PAPER34 = [
    "ta01", "ta02", "ta03", "ta04", "ta05", "ta06", "ta07", "ta08", "ta09",
    "ta10", "ta11", "ta12", "ta13", "ta21", "ta22",
    "abz5", "abz6", "ft06", "ft10", "ft20", "la03", "la04", "la06", "la07",
    "la08", "la09", "la10", "orb01", "orb02", "orb03", "swv01", "swv03",
    "swv04", "swv05",
]


def lower_bound(env):
    return max(max(env.machine_free), env.time)



def run_beam(agent, env, width, max_states=500000):
    env.reset()
    beams = [env]
    steps = 0
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
            logits, _, _ = agent.forward(e, available=avail)
            order = torch.argsort(logits, descending=True).tolist()
            for rank in order[:width]:
                ne = copy.deepcopy(e)
                ne.step(avail[rank])
                next_beams.append(ne)
        next_beams.sort(key=lower_bound)
        beams = next_beams[:width]
        steps += 1
    return min(e.get_makespan() for e in beams), steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hetero_model.pt")
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--out", default="")
    parser.add_argument("--instances", nargs="*", default=None)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    agent = HeteroPolicy(hidden_dim=64, n_layers=2, n_max_jobs=30).eval()
    agent.load_state_dict(torch.load(os.path.join(here, args.model), map_location="cpu"))

    best_path = os.path.join(here, "jsplib_instances.json")
    best_data = {rec["name"]: rec for rec in json.load(open(best_path, encoding="utf-8"))}
    reference = {}
    ref_path = os.path.join(here, "results_official_corrected.json")
    if os.path.exists(ref_path):
        reference = json.load(open(ref_path, encoding="utf-8"))

    if args.instances is None:
        instances = PAPER34
    else:
        instances = args.instances

    root = os.path.join(here, "taillard")
    bench_root = os.path.join(here, "benchmarks")
    results = {}
    for name in instances:
        if os.path.exists(os.path.join(root, name)):
            path = os.path.join(root, name)
        else:
            path = os.path.join(bench_root, name)
        n_jobs, n_mach, ops = parse_jsplib(path)
        env = JSPEnv.from_operations(n_jobs, n_mach, ops)
        t0 = time.time()
        mk, steps = run_beam(agent, env, args.width)
        bks = best_data.get(name, {}).get("optimum")
        if bks is None:
            bks = best_data.get(name, {}).get("bounds", {}).get("upper")
        ref = reference.get(name, {})
        gap = (mk - bks) / bks * 100 if bks else None
        results[name] = {
            "beam_mk": float(mk),
            "greedy_mk": float(ref.get("hgmn", float("nan"))),
            "spt": float(ref.get("spt", float("nan"))),
            "best_rule": float(ref.get("best_rule", float("nan"))),
            "known_best": bks,
            "beam_gap_bks_pct": gap,
            "steps": steps,
            "seconds": time.time() - t0,
        }
        print(
            f"{name}: greedy={results[name]['greedy_mk']:.0f} beam={mk:.0f} "
            f"bks={bks} gap={gap:.1f}% time={results[name]['seconds']:.1f}s",
            flush=True,
        )

    if args.out:
        with open(os.path.join(here, args.out), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
