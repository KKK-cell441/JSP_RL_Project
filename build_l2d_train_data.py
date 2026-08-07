#!/usr/bin/env python3
"""Build HGNN training states from official L2D teacher schedules."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jsp_env import JSPEnv
from hetero_model import HeteroPolicy


def collect_from_teacher(record, agent):
    dur = record["dur"]
    mch = record["mch"]
    n_jobs = len(dur)
    n_mach = len(dur[0])
    ops = [
        [(int(mch[j][k]) - 1, float(dur[j][k])) for k in range(n_mach)]
        for j in range(n_jobs)
    ]
    env = JSPEnv.from_operations(n_jobs, n_mach, ops)
    schedule = {
        (item["job"], item["op_idx"]): item["start"]
        for item in record["schedule"]
    }
    env.reset()
    entries = []
    while not env.done:
        avail = env._get_available_operations()
        if not avail:
            env._advance_time()
            continue
        best = min(avail, key=lambda j: schedule[(j, env.job_progress[j])])
        job_feats = agent.get_job_features(env)
        machine_feats = agent.get_machine_features(env)
        je, me, jme = agent.build_edges(env)
        entries.append((
            job_feats.clone(),
            machine_feats.clone(),
            je.clone(),
            me.clone(),
            jme.clone(),
            int(best),
            list(avail),
        ))
        env.step(best)
    return entries


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    agent = HeteroPolicy(hidden_dim=64, n_layers=2, n_max_jobs=30).eval()
    files = [
        os.path.join(here, "L2D_official", "l2d_teacher_15x15.json"),
        os.path.join(here, "L2D_official", "l2d_teacher_20x15.json"),
        os.path.join(here, "L2D_official", "l2d_teacher_20x20.json"),
    ]
    entries = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            ep = collect_from_teacher(rec, agent)
            entries.extend(ep)
            print(f"{rec['instance']}: states={len(ep)} mk={rec['makespan']}", flush=True)
    out_path = os.path.join(here, "hetero_train_data_l2d.pt")
    torch.save(entries, out_path)
    print(f"Saved {len(entries)} states to {out_path}")


if __name__ == "__main__":
    main()
