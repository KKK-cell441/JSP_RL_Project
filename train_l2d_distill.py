#!/usr/bin/env python3
"""Train a homogeneous L2D-style baseline on the same teacher-distillation states.

This isolates the contribution of the heterogeneous graph representation under
identical teacher labels and the true available-action mask.
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from ppo_agent import PPOAgent


def build_edges(n_jobs):
    edges = [(i, j) for i in range(n_jobs) for j in range(n_jobs) if i != j]
    if not edges:
        edges = [(0, 0)]
    return torch.LongTensor(edges).t()


def forward_homogeneous(agent, job_feats, available):
    device = next(agent.parameters()).device
    x = job_feats.unsqueeze(0).to(device)
    edges = build_edges(job_feats.size(0)).to(device)
    h = agent.encoder(x, edges)
    actor_h = F.relu(agent.actor(h.squeeze(0)))
    logits = agent.actor_head(actor_h).squeeze(-1)
    if available:
        avail_t = torch.LongTensor(available).to(device)
        logits = logits[avail_t]
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="hetero_train_data_l2d.pt")
    parser.add_argument("--out", default="l2d_homogeneous_l2d.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data = torch.load(os.path.join(here, args.data), map_location="cpu", weights_only=False)
    print(f"Loaded {len(data)} states")

    agent = PPOAgent(feat_dim=4, hidden_dim=64, n_layers=2)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
    rng = random.Random(42)
    loss_hist = []
    t0 = time.time()
    for epoch in range(args.epochs):
        idx = list(range(len(data)))
        rng.shuffle(idx)
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(idx), args.batch):
            batch_idx = idx[start : start + args.batch]
            optimizer.zero_grad()
            loss = torch.tensor(0.0)
            for i in batch_idx:
                job_feats, _, _, _, _, action, avail = data[i]
                if action not in avail:
                    avail = list(range(job_feats.size(0)))
                    target = torch.tensor([int(action)])
                else:
                    target = torch.tensor([avail.index(action)])
                logits = forward_homogeneous(agent, job_feats, avail)
                loss = loss + F.cross_entropy(logits.unsqueeze(0), target)
            loss = loss / len(batch_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        loss_hist.append(avg)
        print(f"epoch {epoch + 1}/{args.epochs} loss={avg:.4f} elapsed={time.time() - t0:.1f}s", flush=True)

    out_path = os.path.join(here, args.out)
    torch.save(agent.state_dict(), out_path)
    with open(out_path.replace(".pt", "_loss.json"), "w", encoding="utf-8") as f:
        json.dump({"epochs": args.epochs, "batch_size": args.batch, "loss_hist": loss_hist}, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
