"""
Heterogeneous Graph Neural Network policy for JSS.

Extends the job-level GAT with machine nodes and operation-machine edges.
Graph: [job nodes (n) | machine nodes (m)] with:
  - job-job edges (full connectivity)
  - job-machine edges (which machine each job's current operation needs)
  - machine-machine edges (optional full connectivity)
Variable-size graphs are constructed directly from the current instance, so
inference does not require padding to a fixed topology.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HeteroGATLayer(nn.Module):
    """Graph attention layer supporting multiple node types via type-specific projections."""

    def __init__(self, in_dim, out_dim, n_types=2, dropout=0.1):
        super().__init__()
        self.out_dim = out_dim
        # Type-specific projections
        self.W = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(n_types)])
        # Attention projection (shared)
        self.a = nn.Linear(out_dim * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, node_types):
        """
        x: [N, in_dim] node features
        edge_index: [2, E] edges
        node_types: [N] int tensor, 0=job, 1=machine
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        # Type-specific projection
        h = torch.zeros_like(x)
        for t in range(self.W.__len__()):
            mask = (node_types == t)
            if mask.any():
                h[mask] = self.W[t](x[mask])
        # Attention
        h_src = h[src]  # [E, D]
        h_dst = h[dst]
        attn = self.leaky_relu(self.a(torch.cat([h_src, h_dst], dim=-1))).squeeze(-1)  # [E]
        attn = torch.exp(attn - attn.max())
        # Normalize per destination
        zeros = torch.zeros(N, device=x.device)
        attn_sum = zeros.scatter_add(0, dst, attn)
        attn_norm = attn / (attn_sum[dst] + 1e-8)
        # Aggregate
        out = torch.zeros(N, self.out_dim, device=x.device)
        weighted = attn_norm.unsqueeze(-1) * h_src
        out = out.scatter_add(0, dst.unsqueeze(-1).expand(-1, self.out_dim), weighted)
        return out


class HeteroGNN(nn.Module):
    """Heterogeneous GNN encoder with job and machine nodes."""

    def __init__(self, job_feat_dim=4, machine_feat_dim=3, hidden_dim=64, n_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.job_embed = nn.Linear(job_feat_dim, hidden_dim)
        self.machine_embed = nn.Linear(machine_feat_dim, hidden_dim)
        self.layers = nn.ModuleList([HeteroGATLayer(hidden_dim, hidden_dim, n_types=2) for _ in range(n_layers)])
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, job_feats, machine_feats, job_edges, machine_edges, job_machine_edges):
        """
        job_feats: [n_jobs, job_feat_dim]
        machine_feats: [n_machines, machine_feat_dim]
        job_edges: [2, E_jj]
        machine_edges: [2, E_mm]
        job_machine_edges: [2, E_jm]
        """
        n_j = job_feats.size(0)
        n_m = machine_feats.size(0)
        N = n_j + n_m
        # Type embeddings
        h_job = F.relu(self.job_embed(job_feats))
        h_mach = F.relu(self.machine_embed(machine_feats))
        x = torch.cat([h_job, h_mach], dim=0)  # [N, D]
        node_types = torch.cat([torch.zeros(n_j, dtype=torch.long), torch.ones(n_m, dtype=torch.long)]).to(job_feats.device)

        # Build full edge set
        edges = []
        if job_edges.numel() > 0:
            edges.append(job_edges)
        if machine_edges.numel() > 0:
            edges.append(machine_edges)
        if job_machine_edges.numel() > 0:
            edges.append(job_machine_edges)
        # Add reverse edges
        all_edges = []
        for e in edges:
            all_edges.append(e)
            all_edges.append(torch.stack([e[1], e[0]]))
        edge_index = torch.cat(all_edges, dim=1) if all_edges else torch.zeros(2, 1, dtype=torch.long)

        for layer in self.layers:
            x = layer(x, edge_index, node_types)
            x = F.relu(x)

        x = self.output(x)
        job_out = x[:n_j]
        machine_out = x[n_j:]
        return job_out, machine_out


class HeteroPolicy(nn.Module):
    """Heterogeneous GNN policy: job-level scores with machine context."""

    def __init__(self, job_feat_dim=4, machine_feat_dim=3, hidden_dim=64, n_layers=2, n_max_jobs=10):
        super().__init__()
        self.n_max_jobs = n_max_jobs
        # n_max_jobs is retained for backwards compatibility; forward() builds
        # a graph of the current instance size and does not use padding.
        self.encoder = HeteroGNN(job_feat_dim, machine_feat_dim, hidden_dim, n_layers)
        # Policy head: combine job embedding + machine context
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Critic (for future RL extension)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def get_job_features(self, env):
        """[n_jobs, 4] normalized job features."""
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
        """[n_machines, 3] normalized machine load features."""
        feats = []
        max_t = max(max(env.machine_free), 1.0)
        for m in range(env.n_machines):
            # machine load (free time normalized), busy count, contention
            busy = sum(1 for s in env.schedule if s[1] == m)
            feats.append([env.machine_free[m] / max_t, busy / (max(busy, 1) + 1e-8), 0.0])
        return torch.FloatTensor(feats)

    def build_edges(self, env):
        """Build job-job, machine-machine, job-machine edges."""
        n_j = env.n_jobs
        n_m = env.n_machines
        job_edges = torch.LongTensor([(i, j) for i in range(n_j) for j in range(n_j) if i != j]).t() if n_j > 1 else torch.zeros(2, 0, dtype=torch.long)
        machine_edges = torch.LongTensor([(i, j) for i in range(n_m) for j in range(n_m) if i != j]).t() if n_m > 1 else torch.zeros(2, 0, dtype=torch.long)
        # Job-machine edges: current operation machine for each job
        jm = []
        for j in range(n_j):
            if env.job_progress[j] < len(env.operations[j]):
                m = env.operations[j][env.job_progress[j]].machine
                jm.append((j, n_j + m))
        job_machine_edges = torch.LongTensor(jm).t() if jm else torch.zeros(2, 0, dtype=torch.long)
        return job_edges, machine_edges, job_machine_edges

    def forward_graph(self, job_feats, machine_feats, job_edges, machine_edges, jm_edges, available=None):
        """Score a raw graph state; used by training and evaluation."""
        device = next(self.parameters()).device
        job_feats = job_feats.to(device)
        machine_feats = machine_feats.to(device)
        job_edges = job_edges.to(device)
        machine_edges = machine_edges.to(device)
        jm_edges = jm_edges.to(device)
        job_out, machine_out = self.encoder(job_feats, machine_feats, job_edges, machine_edges, jm_edges)
        machine_context = machine_out.mean(dim=0, keepdim=True).expand(job_out.size(0), -1)
        logits = self.actor(torch.cat([job_out, machine_context], dim=-1)).squeeze(-1)
        value = self.critic(torch.cat([job_out.mean(dim=0), machine_out.mean(dim=0)], dim=-1)).squeeze(-1)
        if available is None:
            available = list(range(job_feats.size(0)))
        if available:
            avail_t = torch.LongTensor(available).to(job_feats.device)
            return logits[avail_t], value, available
        return logits, value, available

    def forward(self, env, available=None, mask_padded=True):
        """Return policy logits for available jobs + state value."""
        job_feats = self.get_job_features(env)
        machine_feats = self.get_machine_features(env)
        device = next(self.parameters()).device
        job_feats = job_feats.to(device)
        machine_feats = machine_feats.to(device)
        job_edges, machine_edges, jm_edges = self.build_edges(env)
        job_edges = job_edges.to(device)
        machine_edges = machine_edges.to(device)
        jm_edges = jm_edges.to(device)
        job_out, machine_out = self.encoder(job_feats, machine_feats, job_edges, machine_edges, jm_edges)

        # Machine context for each job
        machine_context = machine_out.mean(dim=0, keepdim=True).expand(job_out.size(0), -1)
        combined = torch.cat([job_out, machine_context], dim=-1)
        logits = self.actor(combined).squeeze(-1)  # [n_jobs]

        # State value from global pooling
        value = self.critic(torch.cat([job_out.mean(dim=0), machine_out.mean(dim=0)], dim=-1)).squeeze(-1)

        if available is None:
            available = env._get_available_operations()
        if available:
            avail_t = torch.LongTensor(available).to(job_feats.device)
            return logits[avail_t], value, available
        return logits, value, available

    def get_action(self, env, deterministic=True):
        logits, value, available = self.forward(env)
        if not available:
            return None, None, value
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            idx = probs.argmax().item()
        else:
            idx = torch.multinomial(probs, 1).item()
        return available[idx], None, value


def pad_job_features(feats, n_max):
    """Legacy helper; forward() builds variable-size graphs directly."""
    n = feats.size(0)
    out = torch.zeros(n_max, feats.size(1))
    out[:n] = feats
    return out


def evaluate_hetero(agent, env, device):
    """Evaluate hetero agent on an environment."""
    agent.eval()
    env.reset()
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


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from jsp_env import JSPEnv

    pt = JSPEnv.generate_random_instance(6, 5, seed=42)
    env = JSPEnv(6, 5, pt)
    agent = HeteroPolicy(n_max_jobs=10)
    action, _, value = agent.get_action(env)
    print(f"HeteroPolicy test: action={action}, value={value.item():.3f}")
    print(f"Params: {sum(p.numel() for p in agent.parameters())}")
