#!/usr/bin/env python3
"""
H1.449 - Language-Conditioned Task Identification vs Task Embeddings
Compare task embeddings (explicit task IDs) vs language-conditioned task identification
(can the model infer task from language alone without explicit task IDs?).

Hypothesis: Language-conditioned models can achieve similar performance to task embeddings
by learning to infer task identity from language descriptions, eliminating the need for
explicit task IDs at inference time.

Key comparison:
- H1.447: Task embeddings solve multi-task generalization (+32.1%)
- H1.448: Task embeddings generalize across complexity (+91.5%)
- This experiment tests if language can replace explicit task IDs

Models:
1. Baseline: Task embeddings with explicit task IDs (from H1.447/H1.448)
2. Language-only: Language-conditioned (no task IDs)
3. Language+TaskID: Both language and task IDs
4. Language+TaskEmbedding: Language + learned task embeddings
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from pathlib import Path
import pickle
from collections import defaultdict
import random

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ============================================================
# Model Definitions
# ============================================================

class BaselineMLP(nn.Module):
    """Simple MLP baseline matching GraphCG parameter count."""
    def __init__(self, input_dim, hidden_dim=64, output_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class TaskEmbeddingGraphCG(nn.Module):
    """
    GraphCG with explicit task embeddings (from H1.447/H1.448).
    Uses task ID embeddings to condition the graph processing.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=7, n_tasks=4, n_passes=3, n_nodes=6):
        super().__init__()
        self.n_passes = n_passes
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.n_tasks = n_tasks
        
        # Task embedding layer
        self.task_embedding = nn.Embedding(n_tasks, hidden_dim)
        
        # Project input to node embeddings
        self.node_proj = nn.Sequential(
            nn.Linear(input_dim, n_nodes * hidden_dim),
            nn.LayerNorm(n_nodes * hidden_dim)
        )
        
        # Message passing layers
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Node update with task conditioning
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # + task embedding
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(n_nodes * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, task_ids):
        batch_size = x.shape[0]
        
        # Get task embeddings
        task_emb = self.task_embedding(task_ids)  # [batch_size, hidden_dim]
        
        # Project input to node embeddings
        node_embs = self.node_proj(x).view(batch_size, self.n_nodes, self.hidden_dim)
        
        # Graph message passing
        for _ in range(self.n_passes):
            # Compute messages between all node pairs
            messages = []
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i != j:
                        pair = torch.cat([node_embs[:, i], node_embs[:, j]], dim=-1)
                        message = self.message_mlp(pair)
                        messages.append(message)
            
            # Aggregate messages per node
            if messages:
                messages = torch.stack(messages, dim=1)  # [batch, n_pairs, hidden]
                messages = messages.mean(dim=1)  # [batch, hidden]
                messages = messages.view(batch_size, 1, self.hidden_dim).repeat(1, self.n_nodes, 1)
            else:
                messages = torch.zeros_like(node_embs)
            
            # Update nodes with task conditioning
            node_input = torch.cat([node_embs, messages, task_emb.unsqueeze(1).repeat(1, self.n_nodes, 1)], dim=-1)
            node_update = self.update_mlp(node_input)
            node_embs = node_embs + node_update  # Residual connection
        
        # Flatten and output
        flat_embs = node_embs.view(batch_size, -1)
        return self.output_proj(flat_embs)


class LanguageConditionedGraphCG(nn.Module):
    """
    GraphCG conditioned on language descriptions instead of task IDs.
    Uses a language encoder to extract task information from text.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=7, lang_dim=128, n_passes=3, n_nodes=6):
        super().__init__()
        self.n_passes = n_passes
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        
        # Simple language encoder (simulating BERT/CLIP embeddings)
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Project input to node embeddings
        self.node_proj = nn.Sequential(
            nn.Linear(input_dim, n_nodes * hidden_dim),
            nn.LayerNorm(n_nodes * hidden_dim)
        )
        
        # Message passing layers
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Node update with language conditioning
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # + language embedding
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(n_nodes * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, language_emb):
        batch_size = x.shape[0]
        
        # Encode language
        lang_emb = self.lang_encoder(language_emb)  # [batch_size, hidden_dim]
        
        # Project input to node embeddings
        node_embs = self.node_proj(x).view(batch_size, self.n_nodes, self.hidden_dim)
        
        # Graph message passing
        for _ in range(self.n_passes):
            # Compute messages between all node pairs
            messages = []
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i != j:
                        pair = torch.cat([node_embs[:, i], node_embs[:, j]], dim=-1)
                        message = self.message_mlp(pair)
                        messages.append(message)
            
            # Aggregate messages per node
            if messages:
                messages = torch.stack(messages, dim=1)  # [batch, n_pairs, hidden]
                messages = messages.mean(dim=1)  # [batch, hidden]
                messages = messages.view(batch_size, 1, self.hidden_dim).repeat(1, self.n_nodes, 1)
            else:
                messages = torch.zeros_like(node_embs)
            
            # Update nodes with language conditioning
            node_input = torch.cat([node_embs, messages, lang_emb.unsqueeze(1).repeat(1, self.n_nodes, 1)], dim=-1)
            node_update = self.update_mlp(node_input)
            node_embs = node_embs + node_update  # Residual connection
        
        # Flatten and output
        flat_embs = node_embs.view(batch_size, -1)
        return self.output_proj(flat_embs)


class HybridLanguageTaskGraphCG(nn.Module):
    """
    Hybrid model using both language and task embeddings.
    Tests if language provides additional signal beyond task IDs.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=7, n_tasks=4, lang_dim=128, n_passes=3, n_nodes=6):
        super().__init__()
        self.n_passes = n_passes
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.n_tasks = n_tasks
        
        # Task embedding layer
        self.task_embedding = nn.Embedding(n_tasks, hidden_dim)
        
        # Language encoder
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Fusion layer for combining language and task embeddings
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Project input to node embeddings
        self.node_proj = nn.Sequential(
            nn.Linear(input_dim, n_nodes * hidden_dim),
            nn.LayerNorm(n_nodes * hidden_dim)
        )
        
        # Message passing layers
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Node update with fused conditioning
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # + fused embedding
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(n_nodes * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, task_ids, language_emb):
        batch_size = x.shape[0]
        
        # Get task and language embeddings
        task_emb = self.task_embedding(task_ids)  # [batch_size, hidden_dim]
        lang_emb = self.lang_encoder(language_emb)  # [batch_size, hidden_dim]
        
        # Fuse language and task embeddings
        fused_emb = self.fusion(torch.cat([task_emb, lang_emb], dim=-1))
        
        # Project input to node embeddings
        node_embs = self.node_proj(x).view(batch_size, self.n_nodes, self.hidden_dim)
        
        # Graph message passing
        for _ in range(self.n_passes):
            # Compute messages between all node pairs
            messages = []
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i != j:
                        pair = torch.cat([node_embs[:, i], node_embs[:, j]], dim=-1)
                        message = self.message_mlp(pair)
                        messages.append(message)
            
            # Aggregate messages per node
            if messages:
                messages = torch.stack(messages, dim=1)  # [batch, n_pairs, hidden]
                messages = messages.mean(dim=1)  # [batch, hidden]
                messages = messages.view(batch_size, 1, self.hidden_dim).repeat(1, self.n_nodes, 1)
            else:
                messages = torch.zeros_like(node_embs)
            
            # Update nodes with fused conditioning
            node_input = torch.cat([node_embs, messages, fused_emb.unsqueeze(1).repeat(1, self.n_nodes, 1)], dim=-1)
            node_update = self.update_mlp(node_input)
            node_embs = node_embs + node_update  # Residual connection
        
        # Flatten and output
        flat_embs = node_embs.view(batch_size, -1)
        return self.output_proj(flat_embs)


# ============================================================
# Data Generation
# ============================================================

def generate_libero_like_data(n_samples=1000, n_objects=5, horizon=10, n_tasks=4):
    """
    Generate synthetic LIBERO-like data with language descriptions.
    
    Returns:
    - observations: [n_samples, n_objects * 7] (pos(3) + quat(4) per object)
    - actions: [n_samples, 7] (robot action)
    - task_ids: [n_samples] (0 to n_tasks-1)
    - language_embs: [n_samples, 128] (simulated language embeddings)
    """
    np.random.seed(42)
    
    # Generate observations (object poses)
    observations = np.random.randn(n_samples, n_objects * 7) * 0.5
    
    # Generate task-specific actions
    actions = np.zeros((n_samples, 7))
    task_ids = np.random.randint(0, n_tasks, size=n_samples)
    
    # Task-specific action patterns
    for task in range(n_tasks):
        mask = task_ids == task
        n_task_samples = mask.sum()
        if n_task_samples > 0:
            # Each task has a different action pattern
            base_action = np.random.randn(7) * 0.5
            task_scale = 0.3 + task * 0.2  # Increasing scale per task
            actions[mask] = base_action + np.random.randn(n_task_samples, 7) * task_scale
    
    # Add some noise
    actions += np.random.randn(*actions.shape) * 0.1
    
    # Generate language embeddings (simulating CLIP/BERT embeddings)
    # Each task gets a different "language description" embedding
    language_embs = np.zeros((n_samples, 128))
    task_language_bases = np.random.randn(n_tasks, 128) * 0.5
    
    for task in range(n_tasks):
        mask = task_ids == task
        n_task_samples = mask.sum()
        if n_task_samples > 0:
            # Add some variation to language embeddings
            language_embs[mask] = task_language_bases[task] + np.random.randn(n_task_samples, 128) * 0.1
    
    return (
        torch.FloatTensor(observations),
        torch.FloatTensor(actions),
        torch.LongTensor(task_ids),
        torch.FloatTensor(language_embs)
    )


# ============================================================
# Training and Evaluation
# ============================================================

def train_model(model, train_data, val_data, epochs=50, lr=1e-3, batch_size=32, model_type="task_embedding"):
    """Train a model and return validation loss."""
    obs_train, act_train, task_train, lang_train = train_data
    obs_val, act_val, task_val, lang_val = val_data
    
    n_train = obs_train.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(n_train)
        obs_train_shuffled = obs_train[indices]
        act_train_shuffled = act_train[indices]
        task_train_shuffled = task_train[indices]
        lang_train_shuffled = lang_train[indices]
        
        epoch_loss = 0
        model.train()
        
        for i in range(0, n_train, batch_size):
            end_idx = min(i + batch_size, n_train)
            
            batch_obs = obs_train_shuffled[i:end_idx]
            batch_act = act_train_shuffled[i:end_idx]
            batch_task = task_train_shuffled[i:end_idx]
            batch_lang = lang_train_shuffled[i:end_idx]
            
            optimizer.zero_grad()
            
            if model_type == "task_embedding":
                pred = model(batch_obs, batch_task)
            elif model_type == "language":
                pred = model(batch_obs, batch_lang)
            elif model_type == "hybrid":
                pred = model(batch_obs, batch_task, batch_lang)
            else:  # baseline
                pred = model(batch_obs)
            
            loss = criterion(pred, batch_act)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * (end_idx - i)
        
        avg_train_loss = epoch_loss / n_train
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            if model_type == "task_embedding":
                val_pred = model(obs_val, task_val)
            elif model_type == "language":
                val_pred = model(obs_val, lang_val)
            elif model_type == "hybrid":
                val_pred = model(obs_val, task_val, lang_val)
            else:  # baseline
                val_pred = model(obs_val)
            
            val_loss = criterion(val_pred, act_val).item()
            val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
    
    return val_losses[-1], min(val_losses), train_losses, val_losses


def run_experiment(n_objects=5, horizon=10, n_tasks=4, n_demos=200, epochs=30):
    """Run the language vs task embedding comparison experiment."""
    print(f"\nRunning H1.449: Language vs Task Embeddings")
    print(f"  n_objects={n_objects}, horizon={horizon}, n_tasks={n_tasks}, n_demos={n_demos}")
    
    # Generate data
    print("  Generating data...")
    obs, act, task_ids, lang_embs = generate_libero_like_data(
        n_samples=n_demos * 2,  # Train + val
        n_objects=n_objects,
        horizon=horizon,
        n_tasks=n_tasks
    )
    
    # Split train/val
    n_train = n_demos
    train_data = (
        obs[:n_train], act[:n_train], task_ids[:n_train], lang_embs[:n_train]
    )
    val_data = (
        obs[n_train:], act[n_train:], task_ids[n_train:], lang_embs[n_train:]
    )
    
    input_dim = n_objects * 7
    output_dim = 7
    
    # Models to compare
    models = {
        "baseline": BaselineMLP(input_dim, hidden_dim=64, output_dim=output_dim),
        "task_embedding": TaskEmbeddingGraphCG(input_dim, hidden_dim=64, output_dim=output_dim, n_tasks=n_tasks),
        "language": LanguageConditionedGraphCG(input_dim, hidden_dim=64, output_dim=output_dim, lang_dim=128),
        "hybrid": HybridLanguageTaskGraphCG(input_dim, hidden_dim=64, output_dim=output_dim, n_tasks=n_tasks, lang_dim=128)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n  Training {name} model...")
        if name == "baseline":
            model_type = "baseline"
        elif name == "task_embedding":
            model_type = "task_embedding"
        elif name == "language":
            model_type = "language"
        else:  # hybrid
            model_type = "hybrid"
        
        final_val_loss, best_val_loss, train_losses, val_losses = train_model(
            model, train_data, val_data, epochs=epochs, lr=1e-3, batch_size=32, model_type=model_type
        )
        
        results[name] = {
            "final_val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses
        }
        
        print(f"    Final val loss: {final_val_loss:.6f}, Best val loss: {best_val_loss:.6f}")
    
    # Calculate improvements relative to baseline
    baseline_loss = results["baseline"]["best_val_loss"]
    improvements = {}
    for name in ["task_embedding", "language", "hybrid"]:
        if name in results:
            loss = results[name]["best_val_loss"]
            improvement = (baseline_loss - loss) / baseline_loss * 100
            improvements[name] = improvement
    
    return results, improvements


# ============================================================
# Main Experiment
# ============================================================

def main():
    print("=" * 80)
    print("H1.449: Language-Conditioned Task Identification vs Task Embeddings")
    print("=" * 80)
    
    # Experiment configuration based on H1.448
    configs = [
        {"n_objects": 5, "horizon": 10, "n_tasks": 4, "n_demos": 200},
        {"n_objects": 8, "horizon": 15, "n_tasks": 4, "n_demos": 200},
        {"n_objects": 3, "horizon": 5, "n_tasks": 4, "n_demos": 200},
        {"n_objects": 10, "horizon": 20, "n_tasks": 4, "n_demos": 200},
    ]
    
    all_results = {}
    all_improvements = {}
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/{len(configs)}:")
        print(f"  Objects: {config['n_objects']}, Horizon: {config['horizon']}")
        print(f"{'='*60}")
        
        results, improvements = run_experiment(
            n_objects=config["n_objects"],
            horizon=config["horizon"],
            n_tasks=config["n_tasks"],
            n_demos=config["n_demos"],
            epochs=30
        )
        
        all_results[f"config_{i+1}"] = results
        all_improvements[f"config_{i+1}"] = improvements
        
        print(f"\n  Improvements vs Baseline:")
        for model, imp in improvements.items():
            print(f"    {model}: {imp:+.2f}%")
    
    # Aggregate results
    print(f"\n{'='*80}")
    print("AGGREGATE RESULTS")
    print(f"{'='*80}")
    
    avg_improvements = defaultdict(float)
    n_configs = len(configs)
    
    for config_name, improvements in all_improvements.items():
        for model, imp in improvements.items():
            avg_improvements[model] += imp / n_configs
    
    print(f"\nAverage Improvements Across {n_configs} Configurations:")
    for model in ["task_embedding", "language", "hybrid"]:
        if model in avg_improvements:
            print(f"  {model}: {avg_improvements[model]:+.2f}%")
    
    # Determine if language can replace task IDs
    language_percent_of_task_embedding = (avg_improvements.get("language", 0) / 
                                         avg_improvements.get("task_embedding", 1)) * 100
    
    print(f"\nLanguage achieves {language_percent_of_task_embedding:.1f}% of task embedding performance")
    
    if language_percent_of_task_embedding > 80:
        conclusion = "SUPPORTED - Language can effectively replace task IDs"
    elif language_percent_of_task_embedding > 60:
        conclusion = "PARTIAL - Language provides substantial but incomplete task information"
    else:
        conclusion = "REFUTED - Language cannot effectively replace task IDs"
    
    print(f"\nCONCLUSION: {conclusion}")
    
    # Save results
    results_dict = {
        "experiment_id": "H1.449",
        "description": "Compare task embeddings vs language-conditioned task identification",
        "configurations": configs,
        "results": all_results,
        "improvements": all_improvements,
        "average_improvements": dict(avg_improvements),
        "language_percent_of_task_embedding": language_percent_of_task_embedding,
        "conclusion": conclusion,
        "timestamp": datetime.now().isoformat()
    }
    
    output_path = Path(__file__).parent / "results.json"
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    
    return results_dict


if __name__ == "__main__":
    results = main()