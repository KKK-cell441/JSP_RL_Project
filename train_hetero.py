#!/usr/bin/env python3
"""Train the heterogeneous GNN policy from saved CP-SAT expert states.

Input: hetero_train_data.pt (list of (job_feats, machine_feats, job_edges,
machine_edges, job_machine_edges, expert_action)).
Output: hetero_model_retrained.pt and hetero_training_loss.json.
"""
import json
import os
import random
import time
import torch
import torch.nn.functional as F

from hetero_model import HeteroPolicy

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "hetero_train_data.pt")
OUT = os.path.join(HERE, "hetero_model_retrained.pt")
LOSS_OUT = os.path.join(HERE, "hetero_training_loss.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

data = torch.load(DATA, map_location="cpu", weights_only=False)
print(f"Loaded {len(data)} training states")

model = HeteroPolicy(hidden_dim=64, n_layers=2, n_max_jobs=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 30
BATCH = 64
rng = random.Random(42)
loss_hist = []
t0 = time.time()

for epoch in range(EPOCHS):
    idx = list(range(len(data)))
    rng.shuffle(idx)
    total_loss, n_batches = 0.0, 0
    for start in range(0, len(idx), BATCH):
        batch_idx = idx[start:start + BATCH]
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)
        for i in batch_idx:
            job_feats, mach_feats, je, me, jme, action = data[i]
            logits, _, _ = model.forward_graph(
                job_feats, mach_feats, je, me, jme,
                available=list(range(job_feats.size(0))),
            )
            target = torch.tensor([int(action)], device=device)
            loss = loss + F.cross_entropy(logits.unsqueeze(0), target)
        loss = loss / len(batch_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    loss_hist.append(avg_loss)
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch}: loss={avg_loss:.4f} ({time.time()-t0:.0f}s)", flush=True)

torch.save(model.state_dict(), OUT)
with open(LOSS_OUT, "w", encoding="utf-8") as f:
    json.dump({"epochs": EPOCHS, "batch_size": BATCH, "loss_hist": loss_hist,
               "model": OUT}, f, indent=2)
print(f"Saved {OUT}")
