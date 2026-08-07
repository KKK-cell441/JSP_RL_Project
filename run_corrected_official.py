#!/usr/bin/env python3
"""Evaluate the trained HGNN on JSPLIB official instances with correct parsing.

The JSPLIB pair format preserves operation order; this script keeps that order in
both the environment and the CP-SAT model. Results are written to
results_official_corrected.json together with known-best bounds from JSPLIB.
"""
import json
import os
import sys
import torch
import numpy as np
from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\17302\Desktop\JSP_RL_Project")
from jsp_env import JSPEnv, schedule_with_heuristic
from hetero_model import HeteroPolicy


ROOT = r"C:\Users\17302\Desktop\JSP_RL_Project"
WORK = os.path.dirname(os.path.abspath(__file__))
BEST_JSON = os.path.join(WORK, "jsplib_instances.json")


def parse_jsplib(path):
    """Return (n_jobs, n_machines, operation_sequences)."""
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    n_jobs, n_mach = map(int, lines[0].split())
    seqs = []
    for j in range(n_jobs):
        tokens = lines[1 + j].split()
        ops = []
        for k in range(n_mach):
            machine = int(tokens[2 * k])
            duration = int(tokens[2 * k + 1])
            ops.append((machine, duration))
        seqs.append(ops)
    return n_jobs, n_mach, seqs


def solve_cp(ops, n_mach, time_limit=5.0, hint=None):
    """CP-SAT model using JSPLIB operation order."""
    n_jobs = len(ops)
    model = cp_model.CpModel()
    horizon = sum(d for job in ops for _, d in job) * 2
    starts, ends = {}, {}
    machine_ops = [[] for _ in range(n_mach)]
    for j, job in enumerate(ops):
        prev = None
        for k, (m, dur) in enumerate(job):
            s = model.NewIntVar(0, horizon, f"s_{j}_{k}")
            e = model.NewIntVar(0, horizon, f"e_{j}_{k}")
            iv = model.NewIntervalVar(s, dur, e, f"i_{j}_{k}")
            starts[(j, k)] = s
            ends[(j, k)] = e
            machine_ops[m].append(iv)
            if prev is not None:
                model.Add(s >= prev)
            prev = e
    for m in range(n_mach):
        if machine_ops[m]:
            model.AddNoOverlap(machine_ops[m])
    makespan = model.NewIntVar(0, horizon, "mk")
    for e in ends.values():
        model.Add(makespan >= e)
    model.Minimize(makespan)
    if hint:
        for key, val in hint.items():
            if key in starts:
                model.AddHint(starts[key], int(round(val)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 42
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return float(solver.ObjectiveValue()), solver.StatusName(status)
    return None, solver.StatusName(status)


def schedule_hints(env):
    """Extract (job, op_idx) start-time hints from a completed env."""
    hints = {}
    for j, ops in enumerate(env.operations):
        for k, op in enumerate(ops):
            for entry in env.schedule:
                if entry[0] == j and entry[1] == op.machine:
                    hints[(j, k)] = entry[2]
                    break
    return hints


def run_agent(agent, env):
    env.reset()
    agent.eval()
    with torch.no_grad():
        while not env.done:
            avail = env._get_available_operations()
            if not avail:
                env._advance_time()
                continue
            job = agent.get_action(env, deterministic=True)[0]
            if job is None:
                break
            env.step(job)
    return env.get_makespan()


def best_known(inst_name, data):
    rec = data.get(inst_name)
    if not rec:
        return None, None
    if rec.get("optimum") is not None:
        return rec["optimum"], "optimum"
    bounds = rec.get("bounds") or {}
    if bounds.get("upper") is not None:
        return bounds["upper"], "best upper bound"
    return None, None


def collect_instances():
    items = []
    for name in sorted(os.listdir(os.path.join(ROOT, "taillard"))):
        if name.startswith("ta") and os.path.isfile(os.path.join(ROOT, "taillard", name)):
            items.append((name, os.path.join(ROOT, "taillard", name), "Taillard"))
    for name in sorted(os.listdir(os.path.join(ROOT, "benchmarks"))):
        fp = os.path.join(ROOT, "benchmarks", name)
        if os.path.isfile(fp):
            items.append((name, fp, "Other"))
    return items


def main(model_path=None, out_path=None):
    if model_path is None:
        model_path = os.path.join(WORK, "hetero_model.pt")
    if out_path is None:
        out_path = os.path.join(WORK, "results_official_corrected.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    agent = HeteroPolicy(n_max_jobs=30).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    with open(BEST_JSON, encoding="utf-8") as f:
        best_data = json.load(f)
    best_data = {rec["name"]: rec for rec in best_data}

    rules = ["SPT", "FIFO", "LPT", "MWKR", "LWKR"]
    results = {}
    for name, path, family in collect_instances():
        n_jobs, n_mach, ops = parse_jsplib(path)
        env = JSPEnv.from_operations(n_jobs, n_mach, ops)
        agent_mk = run_agent(agent, env)
        rule_mks = {r: schedule_with_heuristic(env, r) for r in rules}
        best_rule = min(rule_mks.values())
        cp_mk, cp_status = solve_cp(ops, n_mach, time_limit=5.0)
        bks, bks_kind = best_known(name, best_data)

        hgnn_gap_bks = ((agent_mk - bks) / bks * 100) if bks else None
        spt_gap_bks = ((rule_mks["SPT"] - bks) / bks * 100) if bks else None
        hgnn_vs_spt = (rule_mks["SPT"] - agent_mk) / rule_mks["SPT"] * 100
        cp_gap_bks = ((cp_mk - bks) / bks * 100) if (bks and cp_mk) else None

        results[name] = {
            "family": family,
            "size": f"{n_jobs}x{n_mach}",
            "hgmn": float(agent_mk),
            "spt": float(rule_mks["SPT"]),
            "best_rule": float(best_rule),
            "rule_values": {r: float(v) for r, v in rule_mks.items()},
            "hgmn_vs_spt_pct": float(hgnn_vs_spt),
            "hgmn_gap_bks_pct": hgnn_gap_bks,
            "spt_gap_bks_pct": spt_gap_bks,
            "known_best": bks,
            "known_best_kind": bks_kind,
            "cp_sat_5s": cp_mk,
            "cp_sat_status": cp_status,
            "cp_gap_bks_pct": cp_gap_bks,
        }
        print(f"{name:8s} {n_jobs}x{n_mach:<4d} HGNN={agent_mk:6.0f} SPT={rule_mks['SPT']:6.0f} "
              f"(+{hgnn_vs_spt:5.1f}%) BKS={bks if bks else '-':<6} gap={hgnn_gap_bks if hgnn_gap_bks is not None else -1:6.1f}% CP={cp_mk if cp_mk else '-':<6} {cp_status}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(model_path, out_path)
