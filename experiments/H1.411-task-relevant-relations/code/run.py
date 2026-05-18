#!/usr/bin/env python3
"""
H1.411: Task-Relevant vs Geometric Relational Structure

Hypothesis: CG benefits require task-relevant relational structure (affordances, 
goal-dependent relations), not just geometric relations (distance, contact).

Test approach:
1. Generate datasets with varying degrees of task-relevant structure:
   - Pure geometric: distance, contact, relative position
   - Pure task-relevant: can_pick, can_place_on, is_near_goal
   - Mixed: both types
2. Compare CG vs baseline performance on each
3. If hypothesis correct: CG wins on task-relevant, loses on geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Configuration
CONFIG = {
    "n_train": 400,
    "n_val": 100,
    "epochs": 30,
    "lr": 1e-4,
    "seq_len": 5,
    "n_objects": 3,
    "relation_types": ["geometric", "task_relevant", "mixed"],
}

class RelationDataset(torch.utils.data.Dataset):
    """Dataset with controllable relational structure."""
    
    def __init__(self, n_samples, n_objects, relation_type, seq_len):
        self.n_samples = n_samples
        self.n_objects = n_objects
        self.relation_type = relation_type
        self.seq_len = seq_len
        self.obj_dim = 13
        self.data = self._generate_data()
        
    def _generate_data(self):
        data = []
        
        for _ in range(self.n_samples):
            positions = np.random.randn(self.n_objects, 3) * 0.5
            velocities = np.random.randn(self.n_objects, 3) * 0.1
            
            obj_types = np.zeros((self.n_objects, 4))
            for i in range(self.n_objects):
                obj_types[i, np.random.randint(0, 4)] = 1
            
            obj_colors = np.zeros((self.n_objects, 3))
            for i in range(self.n_objects):
                obj_colors[i, np.random.randint(0, 3)] = 1
            
            trajectory = []
            for t in range(self.seq_len):
                positions = positions + velocities * 0.1 + np.random.randn(self.n_objects, 3) * 0.01
                
                obj_state = np.concatenate([positions, velocities, obj_types, obj_colors], axis=1)
                relations = self._generate_relations(positions, obj_types, t)
                
                trajectory.append({
                    "objects": obj_state,
                    "relations": relations,
                })
            
            target = positions.flatten()
            
            data.append({
                "trajectory": trajectory,
                "target": target,
            })
        
        return data
    
    def _generate_relations(self, positions, obj_types, timestep):
        n_obj = positions.shape[0]
        relations = []
        
        for i in range(n_obj):
            for j in range(n_obj):
                if i >= j:
                    continue
                    
                if self.relation_type == "geometric":
                    rel = self._geometric_relations(positions, i, j)
                elif self.relation_type == "task_relevant":
                    rel = self._task_relevant_relations(positions, obj_types, i, j, timestep)
                else:
                    geom = self._geometric_relations(positions, i, j)
                    task = self._task_relevant_relations(positions, obj_types, i, j, timestep)
                    rel = np.concatenate([geom, task])
                
                relations.append(rel)
        
        return np.concatenate(relations) if relations else np.zeros(5)
    
    def _geometric_relations(self, positions, i, j):
        dist = np.linalg.norm(positions[i] - positions[j])
        contact = 1.0 if dist < 0.1 else 0.0
        rel_pos = (positions[j] - positions[i]) / (dist + 1e-6)
        return np.concatenate([[dist, contact], rel_pos])
    
    def _task_relevant_relations(self, positions, obj_types, i, j, timestep):
        type_i = np.argmax(obj_types[i])
        type_j = np.argmax(obj_types[j])
        
        can_pick_i = 1.0 if type_i in [0, 2] else 0.0
        can_contain_j = 1.0 if type_j in [1, 3] else 0.0
        
        goal_phase = (timestep / self.seq_len) > 0.5
        is_near_goal = 1.0 if goal_phase and can_contain_j else 0.0
        
        is_graspable = 1.0 if type_i == 0 else 0.0
        can_stack = 1.0 if (type_i == 2 and type_j == 1) else 0.0
        
        return np.array([can_pick_i, can_contain_j, is_near_goal, is_graspable, can_stack])
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        obj_states = np.array([t["objects"] for t in item["trajectory"]], dtype=np.float32)
        rel_states = np.array([t["relations"] for t in item["trajectory"]], dtype=np.float32)
        target = np.array(item["target"], dtype=np.float32)
        
        return (
            torch.from_numpy(obj_states),
            torch.from_numpy(rel_states),
            torch.from_numpy(target)
        )


class BaselineModel(nn.Module):
    """Baseline: flatten all and predict."""
    
    def __init__(self, obj_dim, n_objects, seq_len, hidden_dim=128):
        super().__init__()
        self.total_dim = obj_dim * n_objects * seq_len
        
        self.fc1 = nn.Linear(self.total_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, n_objects * 3)
        
    def forward(self, obj_states, rel_states=None):
        # obj_states: (batch, seq_len, n_objects, obj_dim)
        batch = obj_states.size(0)
        x = obj_states.view(batch, -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.predictor(x)
        
        return out


class CognitiveGraphModel(nn.Module):
    """CG: separate physical and semantic, with cross-attention."""
    
    def __init__(self, obj_dim, n_objects, n_relations, seq_len, hidden_dim=128):
        super().__init__()
        self.obj_dim = obj_dim
        self.n_objects = n_objects
        self.n_relations = n_relations
        self.seq_len = seq_len
        
        # Physical: position + velocity = 6 dims
        # Semantic: type + color = 7 dims
        self.physical_encoder = nn.Linear(6, 64)
        self.semantic_encoder = nn.Linear(7, 64)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        # Output
        self.fc1 = nn.Linear(128 * n_objects * seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, n_objects * 3)
        
    def forward(self, obj_states, rel_states):
        batch, seq_len, n_objects, obj_dim = obj_states.shape
        
        # Split physical and semantic
        physical = obj_states[..., :6]  # (batch, seq, n_obj, 6)
        semantic = obj_states[..., 6:]  # (batch, seq, n_obj, 7)
        
        # Encode
        physical_enc = self.physical_encoder(physical)  # (batch, seq, n_obj, 64)
        semantic_enc = self.semantic_encoder(semantic)  # (batch, seq, n_obj, 64)
        
        # Combine
        combined = torch.cat([physical_enc, semantic_enc], dim=-1)  # (batch, seq, n_obj, 128)
        
        # Flatten for processing
        x = combined.view(batch, -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.predictor(x)
        
        return out


def train_model(model, train_loader, val_loader, epochs, lr, n_objects):
    """Train model and return final validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0
        for obj_states, rel_states, targets in train_loader:
            optimizer.zero_grad()
            
            outputs = model(obj_states, rel_states)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for obj_states, rel_states, targets in val_loader:
                outputs = model(obj_states, rel_states)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                n_val += 1
        
        val_loss /= max(n_val, 1)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss


def run_experiment():
    """Run the full experiment."""
    print("=" * 60)
    print("H1.411: Task-Relevant vs Geometric Relational Structure")
    print("=" * 60)
    
    results = []
    
    for rel_type in CONFIG["relation_types"]:
        print(f"\n--- Testing relation type: {rel_type} ---")
        
        train_dataset = RelationDataset(
            CONFIG["n_train"], CONFIG["n_objects"], rel_type, CONFIG["seq_len"]
        )
        val_dataset = RelationDataset(
            CONFIG["n_val"], CONFIG["n_objects"], rel_type, CONFIG["seq_len"]
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
        
        n_relations = (CONFIG["n_objects"] * (CONFIG["n_objects"] - 1)) // 2
        
        print("  Training baseline...")
        baseline = BaselineModel(13, CONFIG["n_objects"], CONFIG["seq_len"])
        baseline_loss = train_model(
            baseline, train_loader, val_loader, CONFIG["epochs"], CONFIG["lr"], CONFIG["n_objects"]
        )
        
        print("  Training CG...")
        cg = CognitiveGraphModel(13, CONFIG["n_objects"], n_relations, CONFIG["seq_len"])
        cg_loss = train_model(
            cg, train_loader, val_loader, CONFIG["epochs"], CONFIG["lr"], CONFIG["n_objects"]
        )
        
        improvement = (baseline_loss - cg_loss) / baseline_loss * 100
        cg_wins = cg_loss < baseline_loss
        
        print(f"  Baseline loss: {baseline_loss:.6f}")
        print(f"  CG loss: {cg_loss:.6f}")
        print(f"  Improvement: {improvement:+.2f}%")
        print(f"  CG wins: {cg_wins}")
        
        results.append({
            "relation_type": rel_type,
            "baseline_loss": float(baseline_loss),
            "cg_loss": float(cg_loss),
            "improvement_percent": float(improvement),
            "cg_wins": cg_wins,
        })
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    cg_wins_count = sum(1 for r in results if r["cg_wins"])
    
    for r in results:
        print(f"  {r['relation_type']}: CG improvement = {r['improvement_percent']:+.2f}% (wins={r['cg_wins']})")
    
    print(f"\n  CG win rate: {cg_wins_count}/{len(results)} = {cg_wins_count/len(results)*100:.0f}%")
    
    geometric_result = next(r for r in results if r["relation_type"] == "geometric")
    task_result = next(r for r in results if r["relation_type"] == "task_relevant")
    
    if geometric_result["cg_wins"] == False and task_result["cg_wins"] == True:
        conclusion = "SUPPORTED"
        finding = "CG wins on task-relevant relations but loses on geometric relations, confirming hypothesis."
    elif geometric_result["cg_wins"] == True and task_result["cg_wins"] == False:
        conclusion = "REFUTED"
        finding = "Opposite of hypothesis: CG wins on geometric but loses on task-relevant."
    else:
        conclusion = "INCONCLUSIVE"
        finding = f"Results don't clearly support or refute hypothesis. Geometric: {geometric_result['cg_wins']}, Task: {task_result['cg_wins']}"
    
    print(f"\n  Conclusion: {conclusion}")
    print(f"  Finding: {finding}")
    
    output = {
        "hypothesis": "H1.411",
        "statement": "CG benefits require task-relevant relational structure, not just geometric",
        "date": datetime.now().isoformat(),
        "config": CONFIG,
        "results": results,
        "conclusion": conclusion,
        "key_finding": finding,
    }
    
    with open("experiments/H1.411-task-relevant-relations/results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    return output


if __name__ == "__main__":
    result = run_experiment()
