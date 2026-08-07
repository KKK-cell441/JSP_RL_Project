"""Graph Transformer encoder variant for JSS policy comparison."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from hetero_model import HeteroPolicy, HeteroGNN


class TransformerEncoderBlock(nn.Module):
    """Per-type projected transformer encoder for job + machine nodes."""

    def __init__(self, hidden_dim, n_heads=4, n_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        # Type-specific input projections
        self.job_proj = nn.Linear(4, hidden_dim)
        self.mach_proj = nn.Linear(3, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.type_embed = nn.Parameter(torch.zeros(2, hidden_dim))
        nn.init.xavier_uniform_(self.type_embed)

    def forward(self, job_feats, machine_feats, mask=None):
        n_j = job_feats.size(0)
        n_m = machine_feats.size(0)
        h_job = F.relu(self.job_proj(job_feats)) + self.type_embed[0]
        h_mach = F.relu(self.mach_proj(machine_feats)) + self.type_embed[1]
        x = torch.cat([h_job, h_mach], dim=0).unsqueeze(0)  # [1, n_j+n_m, D]
        # Full attention over all nodes
        out = self.transformer(x).squeeze(0)
        return out[:n_j], out[n_j:]


class TransformerPolicy(nn.Module):
    """Policy using Graph Transformer encoder + MLP head (comparable to HeteroPolicy)."""

    def __init__(self, hidden_dim=64, n_heads=4, n_layers=2, n_max_jobs=10):
        super().__init__()
        self.n_max_jobs = n_max_jobs
        self.encoder = TransformerEncoderBlock(hidden_dim, n_heads, n_layers)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # Reuse feature extraction from HeteroPolicy via composition
    def _feats(self, env):
        tmp = HeteroPolicy(n_max_jobs=self.n_max_jobs)
        return tmp.get_job_features(env), tmp.get_machine_features(env)

    def forward(self, env, available=None):
        device = next(self.parameters()).device
        jf, mf = self._feats(env)
        jf = jf.to(device)
        mf = mf.to(device)
        job_out, mach_out = self.encoder(job_feats=jf, machine_feats=mf)
        mc = mach_out.mean(dim=0, keepdim=True).expand(job_out.size(0), -1)
        logits = self.actor(torch.cat([job_out, mc], dim=-1)).squeeze(-1)
        value = logits.mean()  # dummy critic for compatibility
        if available is None:
            available = env._get_available_operations()
        if available:
            avail_t = torch.LongTensor(available).to(device)
            return logits[avail_t], value, available
        return logits, value, available

    def forward_graph(self, job_feats, machine_feats, available=None):
        device = next(self.parameters()).device
        job_out, mach_out = self.encoder(job_feats=job_feats.to(device), machine_feats=machine_feats.to(device))
        mc = mach_out.mean(dim=0, keepdim=True).expand(job_out.size(0), -1)
        logits = self.actor(torch.cat([job_out, mc], dim=-1)).squeeze(-1)
        value = logits.mean()
        if available is None:
            available = list(range(job_feats.size(0)))
        if available:
            avail_t = torch.LongTensor(available).to(device)
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


def evaluate_policy(policy, env, device):
    policy.eval()
    env.reset()
    with torch.no_grad():
        while not env.done:
            avail = env._get_available_operations()
            if not avail:
                env._advance_time()
                continue
            action = policy.get_action(env, deterministic=True)[0]
            if action is None:
                break
            env.step(action)
    return env.get_makespan()
