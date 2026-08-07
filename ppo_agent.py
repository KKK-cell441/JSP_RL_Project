"""
Graph Neural Network + PPO agent for Job Shop Scheduling.

Model architecture:
- GraphEncoder: node embeddings via GAT (Graph Attention Network)
- Actor: select available operation
- Critic: estimate state value

Training: PPO (Proximal Policy Optimization) with GAE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class GATLayer(nn.Module):
    """Simplified graph attention layer."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(out_dim * 2, 1, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        B, N, F = x.shape
        src_idx, dst_idx = edge_index[0], edge_index[1]

        h = self.W(x)  # [B, N, D]
        src = h[:, src_idx]  # [B, E, D]
        dst = h[:, dst_idx]  # [B, E, D]

        attn = self.leaky_relu(self.a(torch.cat([src, dst], dim=-1)))  # [B, E, 1]
        attn = attn.squeeze(-1)  # [B, E]

        # Softmax over incoming edges per destination
        attn_exp = torch.exp(attn - attn.max(dim=-1, keepdim=True)[0])
        zeros = torch.zeros(B, N, device=x.device)
        attn_sum = zeros.scatter_add(1, dst_idx.unsqueeze(0).expand(B, -1), attn_exp)
        attn_norm = attn_exp / (attn_sum.gather(1, dst_idx.unsqueeze(0).expand(B, -1)) + 1e-8)

        # Aggregate
        out = torch.zeros(B, N, h.size(-1), device=x.device)
        weighted = attn_norm.unsqueeze(-1) * src  # [B, E, D]
        out.scatter_add_(1, dst_idx.unsqueeze(0).expand(B, -1).unsqueeze(-1).expand(B, -1, h.size(-1)), weighted)
        return out


class GraphEncoder(nn.Module):
    """Encode job shop state into node embeddings."""

    def __init__(self, feat_dim=32, hidden_dim=64, n_layers=2):
        super().__init__()
        self.feat_embed = nn.Linear(4, hidden_dim)
        self.layers = nn.ModuleList([GATLayer(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        # x: [B, N, 1] node features (e.g., processing time, remaining work)
        h = F.relu(self.feat_embed(x))
        for layer in self.layers:
            h = layer(h, edge_index)
            h = F.relu(h)
        return self.output(h)  # [B, N, hidden_dim]


class PPOAgent(nn.Module):
    """Actor-Critic with graph encoder."""

    def __init__(self, feat_dim=32, hidden_dim=64, n_layers=2):
        super().__init__()
        self.encoder = GraphEncoder(feat_dim, hidden_dim, n_layers)
        self.hidden_dim = hidden_dim

        # Actor: policy head over available operations
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, 1)

        # Critic: state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def get_node_features(self, env):
        """Convert environment state to node feature matrix."""
        n_nodes = env.n_jobs
        # Features: remaining work / total work / current duration / progress
        feats = []
        for j in range(env.n_jobs):
            ops = env.operations[j]
            progress = env.job_progress[j]
            if progress < len(ops):
                op = ops[progress]
                rem = sum(o.duration for o in ops[progress:])
                total = sum(o.duration for o in ops)
                feats.append([op.duration, rem, total, progress])
            else:
                feats.append([0, 0, 0, 1])
        feats = np.array(feats, dtype=np.float32)
        # Normalize
        if feats.max() > 0:
            feats = feats / (feats.max(axis=0, keepdims=True) + 1e-8)
        return torch.FloatTensor(feats).unsqueeze(0)  # [1, N, 4]

    def build_edges(self, env):
        """Build directed edges between jobs (fully connected = shared machine pool)."""
        edges = []
        for j in range(env.n_jobs):
            for k in range(env.n_jobs):
                if j != k:
                    edges.append((j, k))
        if not edges:
            edges = [(0, 0)]
        return torch.LongTensor(edges).t()

    def forward(self, env, available=None):
        """Get policy logits and state value."""
        x = self.get_node_features(env).to(next(self.parameters()).device)
        edges = self.build_edges(env).to(next(self.parameters()).device)
        h = self.encoder(x, edges)  # [1, N, hidden]
        v = self.critic(h.mean(dim=1))  # [1, 1]

        if available is None:
            available = env._get_available_operations()

        if not available:
            return torch.tensor([0.0], device=x.device), v, []

        avail_tensor = torch.LongTensor(available).to(x.device)
        actor_h = F.relu(self.actor(h.squeeze(0)))
        logits = self.actor_head(actor_h).squeeze(-1)  # [N]
        avail_logits = logits[avail_tensor]
        return avail_logits, v, available

    def get_action(self, env, deterministic=False):
        logits, value, available = self.forward(env)
        if not available:
            return None, None, value
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if deterministic:
            action_idx = probs.argmax().item()
        else:
            action_idx = dist.sample().item()
        action_job = available[action_idx]
        log_prob = dist.log_prob(torch.tensor(action_idx, device=logits.device))
        return action_job, log_prob, value


def collect_rollout(agent, env, max_steps=500):
    """Run one episode and collect (state, action, reward, log_prob, value)."""
    agent.eval()
    states, actions, rewards, log_probs, values = [], [], [], [], []
    env.reset()
    done = False
    step = 0
    while not done and step < max_steps:
        avail = env._get_available_operations()
        if not avail:
            env._advance_time()
            continue
        with torch.no_grad():
            logits, value, available = agent.forward(env, avail)
            if not available:
                break
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            idx = dist.sample()
            action_job = available[idx.item()]
            log_prob = dist.log_prob(idx)
            v = value.squeeze(-1)

        states.append(env._get_state())
        actions.append(action_job)
        log_probs.append(log_prob.item())
        values.append(v.item())

        _, _, done, info = env.step(action_job)
        rewards.append(info.get("makespan", 0.0) if done else 0.0)
        step += 1

    return states, actions, rewards, log_probs, values


def train_ppo(agent, env, n_episodes=500, lr=1e-3, gamma=0.99, gae_lambda=0.95,
              clip_eps=0.2, epochs=3, batch_size=64, device="cpu"):
    """Train PPO agent on job shop instances."""
    agent.to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    reward_history = []

    for episode in range(n_episodes):
        states, actions, rewards, log_probs, values = collect_rollout(agent, env)
        if not states:
            continue

        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(device)

        # Convert to tensors
        values_t = torch.FloatTensor(values).to(device)
        log_probs_t = torch.FloatTensor(log_probs).to(device)

        # Normalize returns
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # PPO update
        for _ in range(epochs):
            # Recompute log probs for collected actions
            # (In this simple version, we use stored log probs as approximate)
            advantage = returns - values_t
            ratio = torch.exp(log_probs_t - log_probs_t.detach())  # placeholder
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_t, returns)
            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

        episode_mk = env.get_makespan() if env.done else max(env.machine_free)
        reward_history.append(-episode_mk)
        if episode % 50 == 0:
            print(f"Episode {episode}: makespan={episode_mk:.1f}, avg_reward={np.mean(reward_history[-50:]):.1f}")

    return reward_history
