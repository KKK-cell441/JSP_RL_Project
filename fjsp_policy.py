"""FJSP-enabled Heterogeneous Policy: joint job + machine option selection.

Extends HeteroPolicy by scoring each (job, option) pair. Available actions are
(job_id, option_id) where option_id indexes the machine alternatives for the
job's current operation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hetero_model import HeteroGNN


class FJSPHeteroPolicy(nn.Module):
    def __init__(self, job_feat_dim=4, machine_feat_dim=3, hidden_dim=64, n_layers=2, n_max_jobs=10):
        super().__init__()
        self.n_max_jobs = n_max_jobs
        self.encoder = HeteroGNN(job_feat_dim, machine_feat_dim, hidden_dim, n_layers)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.option_proj = nn.Linear(2, hidden_dim)  # machine id + duration features

    def get_job_features(self, env):
        feats = []
        for j in range(env.n_jobs):
            ops = env.operations[j]
            prog = env.job_progress[j]
            if prog < len(ops):
                op = ops[prog]
                rem = sum(o.duration for o in ops[prog:])
                total = sum(o.duration for o in ops)
                feats.append([op.duration, rem, total, prog])
            else:
                feats.append([0, 0, 0, 1])
        f = torch.FloatTensor(feats)
        if f.max() > 0:
            f = f / (f.max(dim=0, keepdim=True).values + 1e-8)
        return f

    def get_machine_features(self, env):
        feats = []
        max_t = max(max(env.machine_free), 1.0)
        for m in range(env.n_machines):
            busy = sum(1 for s in env.schedule if s[1] == m)
            feats.append([env.machine_free[m] / max_t, busy / (max(busy, 1) + 1e-8), 0.0])
        return torch.FloatTensor(feats)

    def build_edges(self, env):
        n_j, n_m = env.n_jobs, env.n_machines
        job_edges = torch.LongTensor([(i, j) for i in range(n_j) for j in range(n_j) if i != j]).t() if n_j > 1 else torch.zeros(2, 0, dtype=torch.long)
        machine_edges = torch.LongTensor([(i, j) for i in range(n_m) for j in range(n_m) if i != j]).t() if n_m > 1 else torch.zeros(2, 0, dtype=torch.long)
        jm = []
        for j in range(n_j):
            if env.job_progress[j] < len(env.operations[j]):
                m = env.operations[j][env.job_progress[j]].machine
                jm.append((j, n_j + m))
        jm_edges = torch.LongTensor(jm).t() if jm else torch.zeros(2, 0, dtype=torch.long)
        return job_edges, machine_edges, jm_edges

    def forward_options(self, env):
        """Score each (job, option) pair for FJSP. Returns dict job->[scores]."""
        device = next(self.parameters()).device
        job_feats = self.get_job_features(env).to(device)
        machine_feats = self.get_machine_features(env).to(device)
        je, me, jme = self.build_edges(env)
        je = je.to(device); me = me.to(device); jme = jme.to(device)
        job_out, machine_out = self.encoder(job_feats, machine_feats, je, me, jme)
        machine_context = machine_out.mean(dim=0, keepdim=True).expand(job_out.size(0), -1)
        job_scores = self.actor(torch.cat([job_out, machine_context], dim=-1)).squeeze(-1)

        # Option-level scores: add machine-id and duration features per option
        result = {}
        for j in range(env.n_jobs):
            if env.job_progress[j] < len(env.operations[j]):
                op = env.operations[j][env.job_progress[j]]
                # op is from FJSP env where each "operation" has .machine/.duration of first option
                # For proper FJSP, env.operations[j][prog] should carry alternatives
                opts = getattr(env, "alternatives", None)
                if opts is not None:
                    options = opts[j][env.job_progress[j]]
                    scores = []
                    for (m, dur) in options:
                        opt_feat = torch.FloatTensor([m / max(env.n_machines, 1), dur / (dur + 1e-8)]).to(device)
                        scores.append(job_scores[j].item() + self.option_proj(opt_feat).sum().item())
                    result[j] = scores
                else:
                    result[j] = [job_scores[j].item()]
        return result

    def get_action(self, env):
        """Return (job, option_id) or None."""
        options = self.forward_options(env)
        best = None
        best_score = -1e18
        for j, scores in options.items():
            for oi, sc in enumerate(scores):
                if sc > best_score:
                    best_score = sc
                    best = (j, oi)
        return best


def run_fjsp_policy(agent, env):
    env.reset()
    agent.eval()
    with torch.no_grad():
        while not env.done:
            avail = env._get_available_operations()
            if not avail:
                env._advance_time()
                continue
            # Only select among available jobs
            options = agent.forward_options(env)
            best = None
            best_score = -1e18
            for jj in avail:
                if jj in options:
                    opts = env.alternatives[jj][env.job_progress[jj]]
                    for oii, sc in enumerate(options[jj]):
                        m = opts[oii][0]
                        if env.machine_free[m] <= env.time + 1e-9 and sc > best_score:
                            best_score = sc
                            best = (jj, oii)
            if best is None:
                break
            j, oi = best
            # Set operation to chosen option before stepping
            opts = env.alternatives[j][env.job_progress[j]]
            m, dur = opts[oi]
            op = env.operations[j][env.job_progress[j]]
            op.machine = m
            op.duration = dur
            env.step(j)
    return env.get_makespan(), sum(env.job_done_time) / env.n_jobs
