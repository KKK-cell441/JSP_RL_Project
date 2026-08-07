#!/usr/bin/env python3
"""Train the Graph Transformer policy on the same teacher-distillation states."""
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from transformer_policy import TransformerPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="hetero_train_data_l2d.pt")
    parser.add_argument("--out", default="transformer_policy_l2d.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data = torch.load(os.path.join(here, args.data), map_location="cpu", weights_only=False)
    print(f"Loaded {len(data)} states")

    model = TransformerPolicy(hidden_dim=64, n_heads=4, n_layers=2, n_max_jobs=30)
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
                jf, mf, _, _, _, action, avail = data[i]
                if action not in avail:
                    avail_all = list(range(jf.size(0)))
                    logits, _, _ = model.forward_graph(jf, mf, available=avail_all)
                    target = torch.tensor([int(action)])
                else:
                    logits, _, _ = model.forward_graph(jf, mf, available=avail)
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
        print(f"epoch {epoch + 1}/{args.epochs} loss={avg:.4f} elapsed={time.time()-t0:.1f}s", flush=True)

    out_path = os.path.join(here, args.out)
    torch.save(model.state_dict(), out_path)
    with open(out_path.replace(".pt", "_loss.json"), "w", encoding="utf-8") as f:
        json.dump({"epochs": args.epochs, "batch_size": args.batch, "loss_hist": loss_hist}, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
