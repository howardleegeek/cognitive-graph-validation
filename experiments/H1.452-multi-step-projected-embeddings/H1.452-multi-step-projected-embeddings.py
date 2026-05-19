#!/usr/bin/env python3
"""
H1.452: Multi-step Task Test with Projected Real Embeddings

Hypothesis: Cognitive Graph with projected real embeddings will show 
increased advantage over simple models on multi-step tasks (3+ sub-goals)
compared to single-step tasks.

Context from H1.451:
- CG with 32-dim projection beats simple language model by +8.16%
- Smaller projections work best (32 > 64 > 128 > 256)
- Real embeddings significantly outperform simulated

This test: Verify if CG's graph structure provides more advantage 
on complex multi-step tasks vs simple single-step tasks.
"""

import sys
import os
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src")
from data_loader import LIBERODataset

# Try to import sentence-transformers for real embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("WARNING: sentence-transformers not available, using simulated embeddings")


class SimpleMLP(nn.Module):
    """Baseline MLP with no language conditioning."""
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs):
        return self.net(obs)


class LanguageConditionedModel(nn.Module):
    """Simple cross-attention model (no graph structure)."""
    def __init__(self, obs_dim, action_dim, lang_dim=384, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        self.fusion = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.output = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, obs, lang_emb):
        # obs: (batch, obs_dim), lang_emb: (batch, lang_dim)
        obs_h = self.obs_encoder(obs).unsqueeze(1)  # (batch, 1, hidden)
        lang_h = self.lang_encoder(lang_emb).unsqueeze(1)  # (batch, 1, hidden)
        
        # Cross-attention
        attn_out, _ = self.fusion(lang_h, obs_h, obs_h)
        return self.output(attn_out.squeeze(1))


class CognitiveGraphProjected(nn.Module):
    """Cognitive Graph with projected embeddings (from H1.451)."""
    def __init__(self, obs_dim, action_dim, lang_dim=384, proj_dim=32, hidden_dim=64, max_nodes=10):
        super().__init__()
        # Project language to lower dimension
        self.lang_projector = nn.Linear(lang_dim, proj_dim)
        
        # Physical state encoder (obs_dim -> hidden_dim)
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        
        # Graph structure: nodes = [state, goal, subgoals]
        # Create learnable node embeddings with max nodes
        self.node_type_emb = nn.Embedding(max_nodes, hidden_dim)
        
        # Additional projection for language to hidden_dim
        self.lang_to_hidden = nn.Linear(proj_dim, hidden_dim)
        
        # Graph attention layers
        self.gat1 = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.gat2 = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Output
        self.output = nn.Linear(hidden_dim, action_dim)
        
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
    
    def forward(self, obs, lang_emb, n_subgoals=0):
        # Project language
        lang_proj = self.lang_projector(lang_emb)  # (batch, proj_dim)
        
        # Encode observation
        obs_h = self.obs_encoder(obs)  # (batch, hidden_dim)
        
        # Build graph: [state_node, goal_node, ...subgoal_nodes]
        batch_size = obs.shape[0]
        
        # Node 0: current state
        state_node = obs_h.unsqueeze(1)  # (batch, 1, hidden)
        
        # Node 1: goal (from language)
        goal_node = self.lang_to_hidden(lang_proj).unsqueeze(1)  # (batch, 1, hidden)
        
        # Create node features
        node_features = [state_node, goal_node]
        
        # Add subgoals if multi-step
        for i in range(min(n_subgoals, self.max_nodes - 2)):
            # Subgoal from intermediate language features with variation
            subgoal = obs_h.unsqueeze(1) * (0.5 + 0.1 * i)  # Slight variation
            node_features.append(subgoal)
        
        # Stack nodes: (batch, num_nodes, hidden)
        nodes = torch.cat(node_features, dim=1)
        
        # Node type embeddings (use first num_nodes types)
        num_nodes = nodes.shape[1]
        node_types = torch.arange(num_nodes, device=obs.device).unsqueeze(0).expand(batch_size, -1)
        nodes = nodes + self.node_type_emb(node_types)
        
        # Graph attention
        attn1, _ = self.gat1(nodes, nodes, nodes)
        attn2, _ = self.gat2(attn1, attn1, attn1)
        
        # Readout from all nodes
        graph_out = attn2.mean(dim=1)  # (batch, hidden)
        
        return self.output(graph_out)


def generate_multi_step_data(n_demos, n_steps, n_unique_instructions, seed=42):
    """Generate multi-step task data with sub-goals."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Task templates with sub-goals
    task_templates = [
        ("pick {obj} then place in {container}", ["pick {obj}", "place in {container}"]),
        ("move {obj} to {location} then push to {location2}", ["move to {location}", "push to {location2}"]),
        ("stack {obj1} on {obj2} then grab {obj3}", ["stack on {obj2}", "grab {obj3}"]),
    ]
    
    colors = ["red", "blue", "green", "yellow"]
    objects = ["cube", "block", "plate", "cup"]
    containers = ["basket", "bin", "box"]
    locations = ["left", "right", "center", "front"]
    
    data = []
    
    for i in range(n_demos):
        # Select task
        template, subgoals = task_templates[i % len(task_templates)]
        
        # Fill in template
        color = np.random.choice(colors)
        obj = np.random.choice(objects)
        container = np.random.choice(containers)
        loc = np.random.choice(locations)
        loc2 = np.random.choice([l for l in locations if l != loc])
        
        full_instruction = template.format(obj=obj, container=container, location=loc, location2=loc2, obj1=obj, obj2=objects[(objects.index(obj)+1)%len(objects)], obj3=objects[(objects.index(obj)+2)%len(objects)])
        
        # Generate trajectory with sub-goals
        seq_len = n_steps * len(subgoals)  # Each sub-goal gets n_steps
        obs_dim = 8
        action_dim = 7
        
        # Observations: [ee_pos(3), joint_angles(4), gripper(1)] = 8
        observations = np.random.randn(seq_len, obs_dim).astype(np.float32)
        
        # Actions: [target_ee_pos(3), target_rotation(3), gripper(1)] = 7
        actions = np.random.randn(seq_len, action_dim).astype(np.float32)
        
        # Add sub-goal markers in observations
        for sg_idx in range(len(subgoals)):
            start_idx = sg_idx * n_steps
            # Mark sub-goal boundaries
            observations[start_idx, -1] = 1.0  # Sub-goal marker
        
        data.append({
            "observations": observations,
            "actions": actions,
            "language": full_instruction,
            "n_subgoals": len(subgoals),
            "subgoals": subgoals
        })
    
    return data


def get_embeddings(texts, model=None, device='cpu'):
    """Get embeddings for text instructions."""
    if HAS_SENTENCE_TRANSFORMERS and model is not None:
        with torch.no_grad():
            emb = model.encode(texts, convert_to_tensor=True)
            return emb.cpu().numpy()
    else:
        # Simulated embeddings
        np.random.seed(42)
        return np.random.randn(len(texts), 384).astype(np.float32)


def train_and_evaluate():
    """Main training and evaluation."""
    print("=" * 60)
    print("H1.452: Multi-step Task Test with Projected Real Embeddings")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load sentence-transformer model
    lang_model = None
    if HAS_SENTENCE_TRANSFORMERS:
        print("Loading sentence-transformer model...")
        lang_model = SentenceTransformer('all-MiniLM-L6-v2')
        lang_model.to(device)
    
    # Configuration
    n_demos = 500
    n_unique_instructions = 136
    obs_dim = 8
    action_dim = 7
    real_lang_dim = 384
    proj_dim = 32  # Best from H1.451
    epochs = 50
    batch_size = 32
    lr = 1e-3
    
    # Test configurations: vary number of sub-goals (steps)
    configs = [
        {"n_steps": 1, "name": "single_step"},
        {"n_steps": 3, "name": "three_step"},
        {"n_steps": 5, "name": "five_step"},
    ]
    
    results = {}
    
    for config in configs:
        n_steps = config["n_steps"]
        config_name = config["name"]
        print(f"\n{'='*40}")
        print(f"Testing {config_name} tasks (n_steps={n_steps})")
        print(f"{'='*40}")
        
        # Generate multi-step data
        print(f"Generating {n_demos} demos with {n_steps} steps per sub-goal...")
        data = generate_multi_step_data(n_demos, n_steps, n_unique_instructions)
        
        # Get unique instructions
        unique_instructions = list(set(d["language"] for d in data))
        print(f"Unique instructions: {len(unique_instructions)}")
        
        # Get embeddings
        print("Computing language embeddings...")
        embeddings_dict = {instr: get_embeddings([instr], lang_model)[0] for instr in unique_instructions}
        
        # Prepare training data
        X = []
        y = []
        lang_X = []
        
        for demo in data:
            for t in range(len(demo["observations"])):
                X.append(demo["observations"][t])
                y.append(demo["actions"][t])
                lang_X.append(embeddings_dict[demo["language"]])
        
        X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
        y = torch.tensor(np.array(y), dtype=torch.float32).to(device)
        lang_X = torch.tensor(np.array(lang_X), dtype=torch.float32).to(device)
        
        n_samples = len(X)
        n_train = int(0.8 * n_samples)
        
        # Shuffle
        perm = torch.randperm(n_samples)
        X = X[perm]
        y = y[perm]
        lang_X = lang_X[perm]
        
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        lang_train, lang_val = lang_X[:n_train], lang_X[n_train:]
        
        # Train models
        models = {}
        
        # 1. Baseline (no language)
        print(f"\nTraining Baseline (MLP)...")
        baseline = SimpleMLP(obs_dim, action_dim).to(device)
        opt = torch.optim.Adam(baseline.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for i in range(0, n_train, batch_size):
                xb = X_train[i:i+batch_size]
                yb = y_train[i:i+batch_size]
                opt.zero_grad()
                loss = F.mse_loss(baseline(xb), yb)
                loss.backward()
                opt.step()
        
        with torch.no_grad():
            baseline_val_loss = F.mse_loss(baseline(X_val), y_val).item()
        models["baseline"] = baseline_val_loss
        print(f"  Baseline val loss: {baseline_val_loss:.6f}")
        
        # 2. Simple Language Model (cross-attention)
        print(f"\nTraining Simple Language Model...")
        simple_lang = LanguageConditionedModel(obs_dim, action_dim, real_lang_dim).to(device)
        opt = torch.optim.Adam(simple_lang.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for i in range(0, n_train, batch_size):
                xb = X_train[i:i+batch_size]
                yb = y_train[i:i+batch_size]
                lb = lang_train[i:i+batch_size]
                opt.zero_grad()
                loss = F.mse_loss(simple_lang(xb, lb), yb)
                loss.backward()
                opt.step()
        
        with torch.no_grad():
            simple_lang_val_loss = F.mse_loss(simple_lang(X_val, lang_val), y_val).item()
        models["simple_language"] = simple_lang_val_loss
        print(f"  Simple Language val loss: {simple_lang_val_loss:.6f}")
        
        # 3. Cognitive Graph with projected embeddings
        print(f"\nTraining CG (proj={proj_dim})...")
        cg = CognitiveGraphProjected(obs_dim, action_dim, real_lang_dim, proj_dim, max_nodes=10).to(device)
        opt = torch.optim.Adam(cg.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for i in range(0, n_train, batch_size):
                xb = X_train[i:i+batch_size]
                yb = y_train[i:i+batch_size]
                lb = lang_train[i:i+batch_size]
                opt.zero_grad()
                # For multi-step, pass number of subgoals
                n_subgoals = n_steps - 1  # n_steps steps means n_steps-1 transitions between subgoals
                loss = F.mse_loss(cg(xb, lb, n_subgoals), yb)
                loss.backward()
                opt.step()
        
        with torch.no_grad():
            cg_val_loss = F.mse_loss(cg(X_val, lang_val, n_steps-1), y_val).item()
        models["cg_projected"] = cg_val_loss
        print(f"  CG Projected val loss: {cg_val_loss:.6f}")
        
        # Store results
        results[config_name] = {
            "n_steps": n_steps,
            "baseline_loss": baseline_val_loss,
            "simple_language_loss": simple_lang_val_loss,
            "cg_projected_loss": cg_val_loss,
            "baseline_improvement_pct": 0.0,
            "simple_language_improvement_pct": (baseline_val_loss - simple_lang_val_loss) / baseline_val_loss * 100,
            "cg_improvement_pct": (baseline_val_loss - cg_val_loss) / baseline_val_loss * 100,
            "cg_vs_simple_pct": (simple_lang_val_loss - cg_val_loss) / simple_lang_val_loss * 100,
        }
        
        print(f"\nResults for {config_name}:")
        print(f"  Baseline: {baseline_val_loss:.6f}")
        print(f"  Simple Language: {simple_lang_val_loss:.6f} ({results[config_name]['simple_language_improvement_pct']:+.2f}%)")
        print(f"  CG Projected: {cg_val_loss:.6f} ({results[config_name]['cg_improvement_pct']:+.2f}%)")
        print(f"  CG vs Simple: {results[config_name]['cg_vs_simple_pct']:+.2f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Multi-step Task Results")
    print("=" * 60)
    
    print("\n| Config | Baseline | Simple Lang | CG Projected | CG vs Simple |")
    print("|--------|----------|-------------|--------------|--------------|")
    for config_name, r in results.items():
        print(f"| {config_name:7} | {r['baseline_loss']:.6f} | {r['simple_language_loss']:.6f} | {r['cg_projected_loss']:.6f} | {r['cg_vs_simple_pct']:+.2f}% |")
    
    # Analyze: Does CG advantage increase with task complexity?
    single_step_cg_vs_simple = results["single_step"]["cg_vs_simple_pct"]
    three_step_cg_vs_simple = results["three_step"]["cg_vs_simple_pct"]
    five_step_cg_vs_simple = results["five_step"]["cg_vs_simple_pct"]
    
    print(f"\nCG vs Simple Language Advantage by Complexity:")
    print(f"  Single-step: {single_step_cg_vs_simple:+.2f}%")
    print(f"  Three-step:  {three_step_cg_vs_simple:+.2f}%")
    print(f"  Five-step:   {five_step_cg_vs_simple:+.2f}%")
    
    # Determine conclusion
    advantage_trend = (three_step_cg_vs_simple - single_step_cg_vs_simple) + (five_step_cg_vs_simple - three_step_cg_vs_simple)
    
    if advantage_trend > 5:
        conclusion = "SUPPORTED - CG advantage increases with task complexity"
    elif advantage_trend < -5:
        conclusion = "REFUTED - CG advantage decreases with task complexity"
    else:
        conclusion = "INCONCLUSIVE - CG advantage stable across complexity levels"
    
    print(f"\nConclusion: {conclusion}")
    print(f"Advantage trend: {advantage_trend:+.2f}%")
    
    # Save results
    output = {
        "experiment_id": "H1.452",
        "description": "Multi-step task test with projected real embeddings",
        "config": {
            "n_demos": n_demos,
            "n_unique_instructions": n_unique_instructions,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "real_lang_dim": real_lang_dim,
            "projection_dim": proj_dim,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "results": results,
        "conclusion": conclusion,
        "advantage_trend": advantage_trend,
    }
    
    # Save JSON
    output_path = Path(__file__).parent / "H1.452-results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return output


if __name__ == "__main__":
    output = train_and_evaluate()
