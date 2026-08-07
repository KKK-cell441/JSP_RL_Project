import sys, os, torch, numpy as np, time, json
sys.path.insert(0, r"C:\Users\17302\Documents\Codex\2026-07-22\mcp\work")
from jsp_env import JSPEnv
from ppo_agent import PPOAgent
import torch.nn.functional as F
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"
n_jobs, n_machines = 6, 5

print("Collecting best-rule data...", flush=True)
all_feats, all_actions = [], []
best_rules = Counter()
t0 = time.time()

for ep in range(150):
    pt = JSPEnv.generate_random_instance(n_jobs, n_machines, seed=100 + ep)
    env = JSPEnv(n_jobs, n_machines, pt)
    agent = PPOAgent(feat_dim=4, hidden_dim=64, n_layers=2)
    candidates = []
    for rule in ["FIFO", "SPT", "LPT", "MWKR", "LWKR"]:
        env.reset()
        feats, acts = [], []
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
            feats.append(agent.get_node_features(env).squeeze(0))
            acts.append(job)
            env.step(job)
        candidates.append((env.get_makespan(), rule, feats, acts))
    best_mk, best_rule, feats, acts = min(candidates, key=lambda x: x[0])
    best_rules[best_rule] += 1
    all_feats.extend(feats)
    all_actions.extend(acts)
    if ep % 50 == 0:
        print(f"  {ep} instances, {time.time()-t0:.0f}s", flush=True)

print(f"Collected {len(all_actions)} states in {time.time()-t0:.0f}s")
print(f"Best rules: {dict(best_rules)}")

print("Training (batched)...", flush=True)
agent = PPOAgent(feat_dim=4, hidden_dim=64, n_layers=2).to(device)


def imitation_forward_batch(agent, feats_batch, device):
    x = torch.stack(feats_batch).to(device)
    B, N, _ = x.shape
    h = F.relu(agent.encoder.feat_embed(x))
    edges = torch.LongTensor([(i, j) for i in range(N) for j in range(N) if i != j] or [(0, 0)]).t().to(device)
    for layer in agent.encoder.layers:
        h = layer(h, edges)
        h = F.relu(h)
    h = agent.encoder.output(h)
    actor_h = F.relu(agent.actor(h))
    return agent.actor_head(actor_h).squeeze(-1)


feats_t = torch.stack(all_feats)
actions_t = torch.tensor(all_actions)
n = len(actions_t)
optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
BATCH, EPOCHS = 128, 30
loss_hist = []
t1 = time.time()
for epoch in range(EPOCHS):
    perm = torch.randperm(n)
    total_loss, n_batches = 0, 0
    for i in range(0, n, BATCH):
        idx = perm[i:i + BATCH]
        xb = feats_t[idx].to(device)
        yb = actions_t[idx].to(device)
        logits = imitation_forward_batch(agent, list(xb), device)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    avg = total_loss / n_batches
    loss_hist.append(avg)
    if epoch % 5 == 0:
        print(f"  Epoch {epoch}: loss={avg:.4f} ({time.time()-t1:.0f}s)", flush=True)

print("Evaluating 30 held-out instances...", flush=True)
results = []
for seed in range(1000, 1030):
    pt = JSPEnv.generate_random_instance(n_jobs, n_machines, seed=seed)
    env = JSPEnv(n_jobs, n_machines, pt)
    hs = {}
    for rule in ["FIFO", "SPT", "LPT", "MWKR", "LWKR"]:
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
        hs[rule] = env.get_makespan()
    best_h = min(hs.values())

    agent.eval()
    env.reset()
    with torch.no_grad():
        while not env.done:
            avail = env._get_available_operations()
            if not avail:
                env._advance_time()
                continue
            x = agent.get_node_features(env).to(device)
            logits = imitation_forward_batch(agent, [x.squeeze(0)], device).squeeze(0)
            avail_t = torch.LongTensor(avail).to(device)
            probs = F.softmax(logits[avail_t], dim=-1)
            job = avail[probs.argmax().item()]
            env.step(job)
    mk_agent = env.get_makespan()
    impr = (best_h - mk_agent) / best_h * 100
    results.append({"seed": seed, "agent": mk_agent, "best_h": best_h, "impr": impr})

avg = np.mean([r["impr"] for r in results])
wins = sum(1 for r in results if r["impr"] > 0)
print()
print("=== Results ===")
print(f"Wins: {wins}/30")
print(f"Avg improvement: {avg:.2f}%")
print(f"Agent avg: {np.mean([r['agent'] for r in results]):.1f}")
print(f"Heuristic avg: {np.mean([r['best_h'] for r in results]):.1f}")

out_dir = r"C:\Users\17302\Documents\Codex\2026-07-22\mcp\work"
torch.save(agent.state_dict(), os.path.join(out_dir, "l2d_final.pt"))
json.dump({"avg_impr": float(avg), "wins": wins, "n": 30, "details": results},
          open(os.path.join(out_dir, "results_l2d_v3.json"), "w"), indent=2)
print("Done")
