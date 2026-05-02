"""
H1.105: Multi-Agent Coordination with Attention
==============================================
Test if attention mechanisms improve multi-agent coordination tasks.
Building on H2.12 (+76.7% with graph), test if attention adds further benefit.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Agent:
    position: np.ndarray
    velocity: np.ndarray
    agent_id: int

@dataclass
class MultiAgentEnv:
    n_agents: int
    dim: int = 2
    arena_size: float = 10.0
    
    def reset(self) -> List[Agent]:
        return [
            Agent(
                position=np.random.uniform(-self.arena_size/2, self.arena_size/2, self.dim),
                velocity=np.zeros(self.dim),
                agent_id=i
            )
            for i in range(self.n_agents)
        ]
    
    def step(self, agents: List[Agent], targets: List[np.ndarray]) -> float:
        """Execute coordination step, return reward."""
        total_error = 0.0
        for agent, target in zip(agents, targets):
            error = np.linalg.norm(agent.position - target)
            total_error += error
            agent.velocity = 0.9 * agent.velocity + 0.1 * (target - agent.position)
            agent.position += agent.velocity * 0.1
        return total_error / len(agents)

class AttentionCoordination(nn.Module):
    """Multi-agent with cross-agent attention."""
    
    def __init__(self, n_agents: int, input_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.input_dim = input_dim
        
        # Individual agent encoding
        self.encoder = nn.Linear(input_dim, hidden)
        
        # Cross-agent attention
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        
        # Output prediction
        self.predictor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim)
        )
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        batch, n, d = states.shape  # [batch, n_agents, dim]
        
        # Encode each agent
        h = torch.relu(self.encoder(states))  # [batch, n, hidden]
        
        # Self-attention among agents
        Q = self.query(h)
        K = self.key(h)
        V = self.value(h)
        
        # Attention scores
        hidden = self.encoder.out_features
        attn_scores = torch.bmm(Q, K.transpose(-2, -1)) / np.sqrt(hidden)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Context vector
        context = torch.bmm(attn_weights, V)
        
        # Predict next positions
        predictions = self.predictor(context)
        
        return predictions

class BaselineCoordination(nn.Module):
    """Standard concatenation baseline."""
    
    def __init__(self, n_agents: int, input_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * n_agents, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim * n_agents)
        )
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        batch, n, d = states.shape
        flat = states.reshape(batch, -1)  # [batch, n*d]
        h = self.encoder(flat)
        out = self.predictor(h)
        return out.reshape(batch, n, d)

def generate_trajectory(env: MultiAgentEnv, n_steps: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Generate multi-agent trajectory."""
    agents = env.reset()
    states_list = []
    targets_list = []
    
    for _ in range(n_steps):
        # Each agent targets another agent (coordination task)
        positions = np.array([a.position for a in agents])
        targets = np.roll(positions, 1, axis=0)  # Target next agent
        
        states = np.array([a.position for a in agents])
        states_list.append(states)
        targets_list.append(targets)
        
        env.step(agents, targets)
    
    return np.array(states_list), np.array(targets_list)

def train_comparison():
    """Train and compare attention vs baseline on multi-agent tasks."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_agents_list = [2, 3, 4, 5, 6, 8]
    results = []
    
    for n_agents in n_agents_list:
        print(f"\n=== {n_agents} Agents ===")
        
        # Generate data
        env = MultiAgentEnv(n_agents=n_agents)
        
        train_states, train_targets = generate_trajectory(env, n_steps=50)
        val_states, val_targets = generate_trajectory(env, n_steps=20)
        
        # Convert to tensors
        train_states_t = torch.tensor(train_states, dtype=torch.float32)
        train_targets_t = torch.tensor(train_targets, dtype=torch.float32)
        val_states_t = torch.tensor(val_states, dtype=torch.float32)
        val_targets_t = torch.tensor(val_targets, dtype=torch.float32)
        
        # Train attention model
        attn_model = AttentionCoordination(n_agents=n_agents)
        attn_opt = torch.optim.Adam(attn_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(200):
            attn_opt.zero_grad()
            pred = attn_model(train_states_t)
            loss = criterion(pred, train_targets_t)
            loss.backward()
            attn_opt.step()
        
        # Evaluate
        with torch.no_grad():
            attn_pred = attn_model(val_states_t)
            attn_loss = criterion(attn_pred, val_targets_t).item()
        
        # Train baseline
        base_model = BaselineCoordination(n_agents=n_agents)
        base_opt = torch.optim.Adam(base_model.parameters(), lr=0.001)
        
        for epoch in range(200):
            base_opt.zero_grad()
            pred = base_model(train_states_t)
            loss = criterion(pred, train_targets_t)
            loss.backward()
            base_opt.step()
        
        with torch.no_grad():
            base_pred = base_model(val_states_t)
            base_loss = criterion(base_pred, val_targets_t).item()
        
        improvement = (base_loss - attn_loss) / base_loss * 100 if base_loss > 0 else 0
        results.append((n_agents, base_loss, attn_loss, improvement))
        
        print(f"  Baseline MSE: {base_loss:.6f}")
        print(f"  Attention MSE: {attn_loss:.6f}")
        print(f"  Improvement: {improvement:+.1f}%")
    
    return results

if __name__ == "__main__":
    print("=" * 60)
    print("H1.105: Multi-Agent Coordination with Attention")
    print("=" * 60)
    
    results = train_comparison()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_improvement = 0.0
    for n, base, attn, imp in results:
        print(f"{n} agents: {imp:+.1f}%")
        total_improvement += imp
    
    avg_improvement = total_improvement / len(results)
    print(f"\nAverage: {avg_improvement:+.1f}%")
    
    # Determine status
    if avg_improvement > 10:
        status = "SUPPORTED"
        print(f"\nStatus: {status} — Attention significantly improves multi-agent coordination")
    elif avg_improvement > 0:
        status = "MARGINAL"
        print(f"\nStatus: {status} — Marginal benefit")
    else:
        status = "REFUTED"
        print(f"\nStatus: {status} — Attention doesn't help multi-agent coordination")