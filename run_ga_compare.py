import sys, os, torch, numpy as np, json, time
sys.path.insert(0, r"C:\Users\17302\Documents\Codex\2026-07-22\mcp\work")
from jsp_env import JSPEnv
from hetero_model import HeteroPolicy
from ga_baseline import ga_solve
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
agent = HeteroPolicy(n_max_jobs=10).to(device)
agent.load_state_dict(torch.load(os.path.join(r"C:\Users\17302\Documents\Codex\2026-07-22\mcp\work", "hetero_model.pt"), map_location=device))


def run_hgnn(env):
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


def run_spt(env):
    env.reset()
    while not env.done:
        avail = env._get_available_operations()
        if not avail:
            env._advance_time()
            continue
        ops = env.operations
        prog = env.job_progress
        job = min(avail, key=lambda j: ops[j][prog[j]].duration)
        env.step(job)
    return env.get_makespan()


print("=== HGNN vs SPT vs GA (10x10, 10 instances) ===")
results = []
for seed in range(20000, 20010):
    pt = JSPEnv.generate_random_instance(10, 10, min_dur=1, max_dur=99, seed=seed)
    env = JSPEnv(10, 10, pt)
    hg = run_hgnn(env)
    spt = run_spt(env)
    t0 = time.time()
    ga = ga_solve(pt, 10, 10, pop_size=40, generations=80, seed=seed)
    gt = time.time() - t0
    results.append({"seed": seed, "hgnn": hg, "spt": spt, "ga": ga, "ga_time": gt})
    print(f"seed {seed}: HGNN={hg}, SPT={spt}, GA={ga} ({gt:.1f}s)")

hg_avg = np.mean([r["hgnn"] for r in results])
spt_avg = np.mean([r["spt"] for r in results])
ga_avg = np.mean([r["ga"] for r in results])
ga_time = np.mean([r["ga_time"] for r in results])
wins = sum(r["hgnn"] < r["ga"] for r in results)
impr = (ga_avg - hg_avg) / ga_avg * 100

print()
print(f"HGNN avg: {hg_avg:.1f}")
print(f"SPT avg: {spt_avg:.1f}")
print(f"GA avg: {ga_avg:.1f}")
print(f"GA avg time: {ga_time:.1f}s")
print(f"HGNN beats GA: {wins}/{len(results)}")
print(f"HGNN vs GA impr: {impr:.2f}%")

json.dump(results, open(os.path.join(r"C:\Users\17302\Documents\Codex\2026-07-22\mcp\work", "results_ga_compare.json"), "w"), indent=2)
print("Saved")
