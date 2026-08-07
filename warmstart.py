#!/usr/bin/env python3
"""Warm-start CP-SAT with the corrected HGNN schedule on official Taillard instances."""
import json
import os
import sys
import torch
import numpy as np
from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\17302\Desktop\JSP_RL_Project")
from jsp_env import JSPEnv
from hetero_model import HeteroPolicy
from run_corrected_official import parse_jsplib, solve_cp, schedule_hints, run_agent

WORK = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\17302\Desktop\JSP_RL_Project"
device = "cuda" if torch.cuda.is_available() else "cpu"
agent = HeteroPolicy(n_max_jobs=30).to(device)
agent.load_state_dict(torch.load(os.path.join(WORK, "hetero_model.pt"), map_location=device))
agent.eval()

files = ["ta01", "ta02", "ta03", "ta11", "ta12", "ta13", "ta21", "ta22"]
results = {}
for f in files:
    path = os.path.join(ROOT, "taillard", f)
    if not os.path.exists(path):
        continue
    n_jobs, n_mach, ops = parse_jsplib(path)
    env = JSPEnv.from_operations(n_jobs, n_mach, ops)
    mk = run_agent(agent, env)
    hints = schedule_hints(env)
    no_hint, st_no = solve_cp(ops, n_mach, time_limit=5.0)
    with_hint, st_hint = solve_cp(ops, n_mach, time_limit=5.0, hint=hints)
    impr = (no_hint - with_hint) / no_hint * 100 if no_hint and with_hint else None
    results[f] = {
        "hgnn_makespan": float(mk),
        "cp_no_hint": no_hint,
        "cp_with_hint": with_hint,
        "status_no": st_no,
        "status_hint": st_hint,
        "improvement_pct": impr,
    }
    print(f"{f}: HGNN={mk:.0f}, CP no-hint={no_hint} ({st_no}), CP hint={with_hint} ({st_hint}), "
          f"improvement={impr if impr is not None else float('nan'):.2f}%")

out = os.path.join(WORK, "results_warmstart_corrected.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Saved {out}")
