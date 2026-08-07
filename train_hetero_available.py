#!/usr/bin/env python3
"""Train the heterogeneous GNN from expert states with true available actions.

The saved data is a list of
(job_feats, machine_feats, job_edges, machine_edges, job_machine_edges,
 action, available).
The policy is trained only on the actions that were actually available at each
state, matching inference-time masking.
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from hetero_model import HeteroPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="hetero_train_data_large.pt")
    parser.add_argument("--out", default="hetero_model_large.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init", default="hetero_model.pt")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, args.data)
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    print(f"Loaded {len(data)} states from {data_path}")

    model = HeteroPolicy(hidden_dim=64, n_layers=2, n_max_jobs=30)
    if args.init and os.path.exists(os.path.join(here, args.init)):
        model.load_state_dict(torch.load(os.path.join(here, args.init), map_location="cpu"))
        print(f"Initialized from {args.init}")
    else:
        print("Training from scratch")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
                job_feats, mach_feats, je, me, jme, action, avail = data[i]
                logits, _, _ = model.forward_graph(
                    job_feats, mach_feats, je, me, jme, available=avail
                )
                if action not in avail:
                    # Defensive fallback: keep the old all-actions behavior.
                    avail_all = list(range(job_feats.size(0)))
                    logits, _, _ = model.forward_graph(
                        job_feats, mach_feats, je, me, jme, available=avail_all
                    )
                    target = torch.tensor([int(action)])
                else:
                    target = torch.tensor([avail.index(action)])
                loss = loss + F.cross_entropy(logits.unsqueeze(0), target)
            loss = loss / len(batch_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        loss_hist.append(avg)
        print(
            f"epoch {epoch + 1}/{args.epochs} loss={avg:.4f} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    out_path = os.path.join(here, args.out)
    torch.save(model.state_dict(), out_path)
    loss_path = out_path.replace(".pt", "_loss.json")
    with open(loss_path, "w", encoding="utf-8") as f:
        json.dump(
            {"epochs": args.epochs, "batch_size": args.batch, "loss_hist": loss_hist},
            f,
            indent=2,
        )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
