import sys, os, torch, numpy as np, time, json
sys.path.insert(0, r"C:\Users\17302\Documents\Codex\2026-07-22\mcp\work")
from jsp_env import JSPEnv
from hetero_model import HeteroPolicy
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

agent = HeteroPolicy(n_max_jobs=20).to(device)
agent.load_state_dict(torch.load(os.path.join(r"C:\Users\17302\Documents\Codex\2026-07-22\mcp\work", "hetero_model.pt"), map_location=device))
print("Loaded hetero model")


def run_agent(env):
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


def run_rule(env, rule):
    env.reset()
    while not env.done:
        avail = env._get_available_operations()
        if not avail:
            env._advance_time()
            continue
        ops = env.operations
        prog = env.job_progress
        if rule == "FIFO":
            job = min(avail)
        elif rule == "SPT":
            job = min(avail, key=lambda j: ops[j][prog[j]].duration)
        elif rule == "LPT":
            job = max(avail, key=lambda j: ops[j][prog[j]].duration)
        elif rule == "MWKR":
            job = max(avail, key=lambda j: sum(o.duration for o in ops[j][prog[j]:]))
        else:
            job = min(avail, key=lambda j: sum(o.duration for o in ops[j][prog[j]:]))
        env.step(job)
    return env.get_makespan()


results = {}
for nj, nm, seeds in [(10, 10, range(14000, 14006)), (15, 15, range(14100, 14106)), (20, 15, range(14200, 14204))]:
    a_s, s_s = [], []
    for seed in seeds:
        pt = JSPEnv.generate_random_instance(nj, nm, min_dur=1, max_dur=99, seed=seed)
        env = JSPEnv(nj, nm, pt)
        a_s.append(run_agent(env))
        s_s.append(run_rule(env, "SPT"))
    key = f"{nj}x{nm}"
    results[key] = {
        "agent": float(np.mean(a_s)),
        "spt": float(np.mean(s_s)),
        "impr": float((np.mean(s_s) - np.mean(a_s)) / np.mean(s_s) * 100),
    }
    print(f"{key}: Agent={results[key]['agent']:.1f}, SPT={results[key]['spt']:.1f}, impr={results[key]['impr']:.1f}%")

json.dump(results, open(os.path.join(r"C:\Users\17302\Documents\Codex\2026-07-22\mcp\work", "results_tailard_hetero.json"), "w"))
print("Saved results_tailard_hetero.json")
# NOTE: this script evaluates generated random instances, not official Taillard files.
# Use run_corrected_official.py for the official JSPLIB benchmark suite.
