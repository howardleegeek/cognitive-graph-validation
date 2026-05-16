"""
H1.376 Experiment: External Memory (Attention-based Key-Value Store)
Test whether external memory can help CG handle 3+ step tasks.

Building on H1.375: 2-layer LSTM temporal memory is optimal (+14.0%)
but CG still struggles with 3+ step tasks (H1.371: -106.6%).

Hypothesis: External memory (key-value store with attention) can help
CG maintain state across longer task horizons.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import json
from torch.utils.data import Dataset, DataLoader


class ManipulationDataset(Dataset):
    """Synthetic manipulation dataset with multi-step sequences."""

    def __init__(self, n_samples: int = 1000, n_steps: int = 3):
        self.n_samples = n_samples
        self.n_steps = n_steps
        
        # Object properties per timestep
        self.objects = torch.randn(n_samples, n_steps, 8)
        self.objects[:, :, 4:7] = torch.softmax(self.objects[:, :, 4:7], dim=2)
        
        # Language instruction embeddings
        self.instructions = torch.randn(n_samples, 32)
        
        # Target actions per timestep
        self.actions = torch.randn(n_samples, n_steps, 5)
        self.actions[:, :, 3:5] = torch.sigmoid(self.actions[:, :, 3:5])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "objects": self.objects[idx],
            "instruction": self.instructions[idx],
            "actions": self.actions[idx],
        }


class BaselineWithLSTM(nn.Module):
    """Baseline: LSTM for temporal reasoning."""

    def __init__(self, obj_dim: int = 8, inst_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.obj_encoder = nn.Linear(obj_dim, 64)
        self.inst_encoder = nn.Linear(inst_dim, 64)
        self.lstm = nn.LSTM(128, hidden_dim, num_layers=2, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, 5)

    def forward(self, objects, instruction, hidden=None):
        # objects: (batch, n_steps, obj_dim)
        batch_size, n_steps, _ = objects.shape
        
        # Encode objects
        obj_enc = F.relu(self.obj_encoder(objects))  # (batch, n_steps, 64)
        
        # Expand instruction to match sequence
        inst_exp = self.inst_encoder(instruction).unsqueeze(1).expand(-1, n_steps, -1)
        
        # Concatenate
        x = torch.cat([obj_enc, inst_exp], dim=-1)  # (batch, n_steps, 128)
        
        # LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        # Action prediction
        actions = self.action_head(lstm_out)
        return actions, hidden


class CognitiveGraphWithExternalMemory(nn.Module):
    """CG with external key-value memory for long-range dependencies."""

    def __init__(
        self,
        obj_dim: int = 8,
        inst_dim: int = 32,
        hidden_dim: int = 128,
        memory_size: int = 16,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.num_heads = num_heads
        
        # Encoders
        self.obj_encoder = nn.Linear(obj_dim, 64)
        self.inst_encoder = nn.Linear(inst_dim, 64)
        
        # Physical state encoder (144 dims)
        self.physical_encoder = nn.Linear(64, 72)
        
        # Semantic state encoder (368 dims)
        self.semantic_encoder = nn.Linear(64, 184)
        
        # External memory: key-value store
        self.memory_keys = nn.Parameter(torch.randn(memory_size, hidden_dim))
        self.memory_values = nn.Parameter(torch.randn(memory_size, hidden_dim))
        
        # Query network for memory attention
        self.query_net = nn.Linear(hidden_dim, hidden_dim)
        
        # Multi-head attention for memory retrieval
        self.memory_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        # Temporal memory (2-layer LSTM from H1.375)
        self.temporal_lstm = nn.LSTM(
            hidden_dim * 2, hidden_dim, num_layers=2, batch_first=True
        )
        
        # Action head
        self.action_head = nn.Linear(hidden_dim, 5)
        
        # Projection layer
        self.project = nn.Linear(256, hidden_dim)
        
        # Initialize memory
        self._init_memory()

    def _init_memory(self):
        """Initialize memory with zeros."""
        nn.init.zeros_(self.memory_keys)
        nn.init.zeros_(self.memory_values)

    def forward(self, objects, instruction, hidden=None, memory_state=None):
        batch_size, n_steps, _ = objects.shape
        
        # Encode objects
        obj_enc = F.relu(self.obj_encoder(objects))  # (batch, n_steps, 64)
        
        # Expand instruction
        inst_exp = F.relu(self.inst_encoder(instruction)).unsqueeze(1).expand(-1, n_steps, -1)
        
        # Split into physical and semantic
        physical = self.physical_encoder(obj_enc)  # (batch, n_steps, 72)
        semantic = self.semantic_encoder(inst_exp)  # (batch, n_steps, 184)
        
        # Combine into unified representation
        unified = torch.cat([physical, semantic], dim=-1)  # (batch, n_steps, 256)
        
        # Project to hidden dim
        h = F.relu(self.project(unified))
        
        # Initialize memory state if None
        if memory_state is None:
            memory_keys = self.memory_keys.unsqueeze(0).expand(batch_size, -1, -1).clone()
            memory_values = self.memory_values.unsqueeze(0).expand(batch_size, -1, -1).clone()
            memory_state = {
                "keys": memory_keys,
                "values": memory_values,
                "pointer": 0,
            }
        
        # Process each timestep
        outputs = []
        lstm_hidden = hidden if hidden else (None, None)
        
        for t in range(n_steps):
            h_t = h[:, t:t+1, :]  # (batch, 1, hidden)
            
            # Query memory
            query = self.query_net(h_t)  # (batch, 1, hidden)
            
            # Attention over memory
            attn_out, _ = self.memory_attention(query, memory_state["keys"], memory_state["values"])
            
            # Combine current hidden with memory
            h_t_with_memory = torch.cat([h_t, attn_out], dim=-1)  # (batch, 1, hidden*2)
            
            # Temporal LSTM
            if lstm_hidden[0] is not None:
                lstm_out, lstm_hidden = self.temporal_lstm(h_t_with_memory, lstm_hidden)
            else:
                lstm_out, lstm_hidden = self.temporal_lstm(h_t_with_memory)
            
            # Predict action
            action = self.action_head(lstm_out)
            outputs.append(action)
            
            # Update memory (write key-value pair)
            # Get the last layer hidden state
            new_key = lstm_hidden[0][-1]  # (batch, hidden)
            new_value = lstm_out.squeeze(1)  # (batch, hidden)
            
            # Update memory with circular buffer
            ptr = memory_state["pointer"] % self.memory_size
            memory_state["keys"][:, ptr, :] = new_key
            memory_state["values"][:, ptr, :] = new_value
            memory_state["pointer"] = ptr + 1
        
        outputs = torch.cat(outputs, dim=1)
        return outputs, (lstm_hidden, memory_state)


def train_model(model, train_loader, epochs=50, lr=1e-3, use_cg=False):
    """Train model and return losses."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            objects = batch["objects"]
            instruction = batch["instruction"]
            actions = batch["actions"]
            
            if use_cg:
                outputs, _ = model(objects, instruction)
            else:
                outputs, _ = model(objects, instruction)
            
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
    
    return losses


def evaluate_model(model, test_loader, use_cg=False):
    """Evaluate model and return MSE."""
    model.eval()
    total_mse = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            objects = batch["objects"]
            instruction = batch["instruction"]
            actions = batch["actions"]
            
            if use_cg:
                outputs, _ = model(objects, instruction)
            else:
                outputs, _ = model(objects, instruction)
            
            mse = F.mse_loss(outputs, actions).item()
            total_mse += mse
            n_batches += 1
    
    return total_mse / n_batches


def run_experiment():
    """Run H1.376 experiment: External Memory test."""
    print("=" * 60)
    print("H1.376: External Memory (Key-Value Store) Test")
    print("=" * 60)
    
    # Configurations to test
    configs = [
        {"name": "baseline_lstm", "use_cg": False, "n_steps": 3},
        {"name": "cg_external_memory", "use_cg": True, "n_steps": 3},
        {"name": "baseline_lstm_2step", "use_cg": False, "n_steps": 2},
        {"name": "cg_external_memory_2step", "use_cg": True, "n_steps": 2},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n--- Testing: {config['name']} ---")
        
        # Create dataset
        dataset = ManipulationDataset(n_samples=500, n_steps=config["n_steps"])
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Create model
        if config["use_cg"]:
            model = CognitiveGraphWithExternalMemory(
                obj_dim=8, inst_dim=32, hidden_dim=128, 
                memory_size=16, num_heads=4
            )
        else:
            model = BaselineWithLSTM(obj_dim=8, inst_dim=32, hidden_dim=128)
        
        # Train
        losses = train_model(model, train_loader, epochs=50, lr=1e-3, use_cg=config["use_cg"])
        
        # Evaluate
        mse = evaluate_model(model, test_loader, use_cg=config["use_cg"])
        
        results[config["name"]] = {
            "mse": mse,
            "final_loss": losses[-1],
            "n_steps": config["n_steps"],
            "use_cg": config["use_cg"],
        }
        
        print(f"  MSE: {mse:.6f}")
    
    # Compare CG vs baseline for each step count
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # 3-step tasks
    baseline_3step = results["baseline_lstm"]["mse"]
    cg_3step = results["cg_external_memory"]["mse"]
    improvement_3step = (baseline_3step - cg_3step) / baseline_3step * 100
    
    # 2-step tasks
    baseline_2step = results["baseline_lstm_2step"]["mse"]
    cg_2step = results["cg_external_memory_2step"]["mse"]
    improvement_2step = (baseline_2step - cg_2step) / baseline_2step * 100
    
    print(f"\n3-Step Tasks:")
    print(f"  Baseline MSE: {baseline_3step:.6f}")
    print(f"  CG + External Memory MSE: {cg_3step:.6f}")
    print(f"  Improvement: {improvement_3step:+.1f}%")
    cg_wins_3step = improvement_3step > 0
    print(f"  CG Wins: {cg_wins_3step}")
    
    print(f"\n2-Step Tasks:")
    print(f"  Baseline MSE: {baseline_2step:.6f}")
    print(f"  CG + External Memory MSE: {cg_2step:.6f}")
    print(f"  Improvement: {improvement_2step:+.1f}%")
    cg_wins_2step = improvement_2step > 0
    print(f"  CG Wins: {cg_wins_2step}")
    
    # Determine conclusion
    if cg_wins_3step and cg_wins_2step:
        conclusion = "SUPPORTED"
    elif not cg_wins_3step and not cg_wins_2step:
        conclusion = "REFUTED"
    else:
        conclusion = "PARTIAL_SUPPORT"
    
    # Save results
    output = {
        "experiment_id": "H1.376",
        "description": "External Memory (Key-Value Store) for 3+ step tasks",
        "config": {
            "memory_size": 16,
            "num_heads": 4,
            "hidden_dim": 128,
            "temporal_layers": 2,
        },
        "results": {
            "baseline_3step_mse": baseline_3step,
            "cg_3step_mse": cg_3step,
            "improvement_3step": improvement_3step,
            "cg_wins_3step": cg_wins_3step,
            "baseline_2step_mse": baseline_2step,
            "cg_2step_mse": cg_2step,
            "improvement_2step": improvement_2step,
            "cg_wins_2step": cg_wins_2step,
        },
        "conclusion": conclusion,
        "key_finding": f"External memory {'improves' if cg_wins_3step else 'does not improve'} CG on 3-step tasks ({improvement_3step:+.1f}%)",
    }
    
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nConclusion: {conclusion}")
    print(f"Key finding: {output['key_finding']}")
    
    return output


if __name__ == "__main__":
    run_experiment()
