#!/usr/bin/env python3
"""Train a homogeneous L2D-style baseline on the same CP-SAT expert states.

This baseline uses the PPOAgent graph encoder but ignores machine nodes, so it
is directly comparable with the heterogeneous GNN under identical labels.
"""
import os
import random
import time
import torch
import torch.nn.functional as F
from ppo_agent import PPOAgent

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "hetero_train_data.pt")
OUT = os.path.join(HERE, "l2d_cpsat.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
data = torch.load(DATA, map_location="cpu", weights_only=False)
print(f"Device: {device}; loaded {len(data)} states")

agent = PPOAgent(feat_dim=4, hidden_dim=64, n_layers=2).to(device)
optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)


def imitation_forward_batch(agent, feats_batch, device):
    x = torch.stack(feats_batch).to(device)
    B, N, _ = x.shape
    h = F.relu(agent.encoder.feat_embed(x))
    edges = torch.LongTensor([(i, j) for i in range(N) for j in range(N) if i != j]).t().to(device)
    for layer in agent.encoder.layers:
        h = layer(h, edges)
        h = F.relu(h)
    h = agent.encoder.output(h)
    actor_h = F.relu(agent.actor(h))
    return agent.actor_head(actor_h).squeeze(-1)


EPOCHS = 30
BATCH = 64
rng = random.Random(42)
feats_all = [t[0] for t in data]
actions_all = [int(t[-1]) for t in data]
t0 = time.time()
for epoch in range(EPOCHS):
    idx = list(range(len(data)))
    rng.shuffle(idx)
    total_loss, n_batches = 0.0, 0
    for start in range(0, len(idx), BATCH):
        batch_idx = idx[start:start + BATCH]
        xb = torch.stack([feats_all[i] for i in batch_idx]).to(device)
        yb = torch.tensor([actions_all[i] for i in batch_idx], device=device)
        logits = imitation_forward_batch(agent, list(xb), device)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    avg = total_loss / max(n_batches, 1)
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch}: loss={avg:.4f} ({time.time()-t0:.0f}s)", flush=True)

torch.save(agent.state_dict(), OUT)
print(f"Saved {OUT}")
