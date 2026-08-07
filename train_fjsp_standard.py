#!/usr/bin/env python3
"""Fine-tune the FJSP policy on expert states from standard benchmarks."""
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from fjsp_policy import FJSPHeteroPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="fjsp_train_data_standard.pt")
    parser.add_argument("--init", default="fjsp_policy.pt")
    parser.add_argument("--out", default="fjsp_policy_standard.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data = torch.load(os.path.join(here, args.data), map_location="cpu", weights_only=False)
    print(f"Loaded {len(data)} states")

    model = FJSPHeteroPolicy(hidden_dim=64, n_layers=2, n_max_jobs=30)
    init_path = os.path.join(here, args.init)
    if args.init and os.path.exists(init_path):
        model.load_state_dict(torch.load(init_path, map_location="cpu"))
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
                jf, mf, je, me, jme, action_j, action_oi, opt_feats, avail = data[i]
                job_out, machine_out = model.encoder(jf, mf, je, me, jme)
                machine_context = machine_out.mean(dim=0, keepdim=True).expand(job_out.size(0), -1)
                job_scores = model.actor(torch.cat([job_out, machine_context], dim=-1)).squeeze(-1)
                if action_j in avail:
                    avail_t = torch.LongTensor(avail)
                    job_target = torch.tensor([avail.index(action_j)])
                    job_logits = job_scores[avail_t].unsqueeze(0)
                else:
                    job_logits = job_scores.unsqueeze(0)
                    job_target = torch.tensor([int(action_j)])
                job_loss = F.cross_entropy(job_logits, job_target)
                opt_scores = model.option_proj(opt_feats).sum(dim=-1).unsqueeze(0)
                opt_loss = F.cross_entropy(opt_scores, torch.tensor([int(action_oi)]))
                loss = loss + job_loss + opt_loss
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
