#!/usr/bin/env python3
"""
H1.460: Investigate why CG performs inconsistently on compositional tasks 
(wins on 4 concepts but loses on 2 and 8). Test whether CG's graph structure 
is better suited for certain concept cardinalities.

Version 2: Fixed data scaling issues.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'src'))

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SyntheticCompositionalDataset(Dataset):
    """Synthetic dataset for testing compositional reasoning with varying concept cardinalities."""
    
    def __init__(self, n_samples=10000, n_concepts=4, seq_len=10, input_dim=512, output_dim=256):
        self.n_samples = n_samples
        self.n_concepts = n_concepts
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Generate synthetic data with proper scaling
        # Each concept is represented as a random vector
        self.concept_vectors = torch.randn(n_concepts, input_dim // 2) * 0.1
        
        # Generate input sequences with proper scaling
        self.inputs = torch.randn(n_samples, seq_len, input_dim) * 0.1
        
        # Generate targets based on compositional reasoning
        # Target is a weighted combination of concept activations
        self.targets = torch.zeros(n_samples, output_dim)
        
        # Create projection matrices
        proj_matrices = torch.randn(n_concepts, input_dim // 2, output_dim) * 0.05
        
        for i in range(n_samples):
            # Randomly select which concepts are active (1-3 concepts)
            n_active = torch.randint(1, min(4, n_concepts + 1), (1,)).item()
            active_concepts = torch.randperm(n_concepts)[:n_active]
            
            # Create target as combination of active concepts
            for concept_idx in active_concepts:
                concept_vec = self.concept_vectors[concept_idx]
                proj = proj_matrices[concept_idx]
                self.targets[i] += concept_vec @ proj
            
            # Normalize by number of active concepts
            self.targets[i] = self.targets[i] / n_active
        
        # Add small noise to inputs to make task non-trivial
        self.inputs = self.inputs + torch.randn_like(self.inputs) * 0.01
        
        # Normalize targets to have unit variance
        target_mean = self.targets.mean(dim=0)
        target_std = self.targets.std(dim=0)
        self.targets = (self.targets - target_mean) / (target_std + 1e-8)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class BaselineModel(nn.Module):
    """Simple MLP baseline with concatenation fusion."""
    
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        # Average over sequence dimension
        x = x.mean(dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class CognitiveGraphModel(nn.Module):
    """Cognitive Graph model with GNN message passing."""
    
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=256, n_concepts=4):
        super().__init__()
        self.n_concepts = n_concepts
        
        # Concept embeddings
        self.concept_embeddings = nn.Embedding(n_concepts, hidden_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layer1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gnn_layer2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch_size = x.shape[0]
        
        # Project input
        x_proj = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # Create graph nodes: one per concept
        concept_nodes = self.concept_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Apply attention between concepts and input features
        # Use input features as keys/values, concepts as queries
        attended_concepts, _ = self.attention(
            concept_nodes,  # queries
            x_proj.mean(dim=1, keepdim=True).repeat(1, self.n_concepts, 1),  # keys (averaged input)
            x_proj.mean(dim=1, keepdim=True).repeat(1, self.n_concepts, 1)   # values
        )
        
        # GNN message passing
        # Simple mean aggregation of neighboring concepts
        neighbor_agg = attended_concepts.mean(dim=1, keepdim=True).repeat(1, self.n_concepts, 1)
        
        # Combine with self features
        combined = torch.cat([attended_concepts, neighbor_agg], dim=-1)
        gnn_out = self.relu(self.gnn_layer1(combined))
        gnn_out = self.dropout(gnn_out)
        gnn_out = self.relu(self.gnn_layer2(gnn_out))
        
        # Pool concepts
        pooled = gnn_out.mean(dim=1)
        
        # Output
        output = self.output_proj(pooled)
        return output

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    """Train a model and return validation loss."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    
    return best_val_loss, train_losses, val_losses

def run_experiment(n_concepts_list=[2, 4, 8], n_samples=2000):
    """Run experiment across different concept cardinalities."""
    results = {}
    
    for n_concepts in n_concepts_list:
        print(f"\n{'='*60}")
        print(f"Testing with {n_concepts} concepts")
        print(f"{'='*60}")
        
        # Create datasets
        train_dataset = SyntheticCompositionalDataset(
            n_samples=n_samples, 
            n_concepts=n_concepts,
            seq_len=10,
            input_dim=512,
            output_dim=256
        )
        
        val_dataset = SyntheticCompositionalDataset(
            n_samples=n_samples // 4, 
            n_concepts=n_concepts,
            seq_len=10,
            input_dim=512,
            output_dim=256
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train baseline model
        print(f"\nTraining Baseline model...")
        baseline_model = BaselineModel(input_dim=512, hidden_dim=256, output_dim=256)
        baseline_val_loss, baseline_train_losses, baseline_val_losses = train_model(
            baseline_model, train_loader, val_loader, epochs=100, lr=0.001
        )
        
        # Train Cognitive Graph model
        print(f"\nTraining Cognitive Graph model...")
        cg_model = CognitiveGraphModel(
            input_dim=512, 
            hidden_dim=256, 
            output_dim=256,
            n_concepts=n_concepts
        )
        cg_val_loss, cg_train_losses, cg_val_losses = train_model(
            cg_model, train_loader, val_loader, epochs=100, lr=0.001
        )
        
        # Calculate improvement
        improvement_pct = ((baseline_val_loss - cg_val_loss) / baseline_val_loss) * 100
        
        # Store results
        results[n_concepts] = {
            'baseline': {
                'final_val_loss': baseline_val_loss,
                'best_val_loss': min(baseline_val_losses) if baseline_val_losses else baseline_val_loss,
                'train_losses': baseline_train_losses,
                'val_losses': baseline_val_losses
            },
            'cognitive_graph': {
                'final_val_loss': cg_val_loss,
                'best_val_loss': min(cg_val_losses) if cg_val_losses else cg_val_loss,
                'train_losses': cg_train_losses,
                'val_losses': cg_val_losses
            },
            'improvement_pct': improvement_pct,
            'cg_wins': improvement_pct > 0
        }
        
        print(f"\nResults for {n_concepts} concepts:")
        print(f"  Baseline validation loss: {baseline_val_loss:.6f}")
        print(f"  Cognitive Graph validation loss: {cg_val_loss:.6f}")
        print(f"  Improvement: {improvement_pct:.2f}%")
        print(f"  CG wins: {improvement_pct > 0}")
    
    return results

def main():
    print("H1.460: Concept Cardinality Investigation (Version 2)")
    print("Testing CG performance across different concept cardinalities (2, 4, 8)")
    print("=" * 80)
    
    # Run experiment
    results = run_experiment(n_concepts_list=[2, 4, 8], n_samples=2000)
    
    # Save results
    output_dir = Path(__file__).parent
    output_path = output_dir / 'results_v2.json'
    
    with open(output_path, 'w') as f:
        json.dump({
            'experiment_id': 'H1.460',
            'description': 'Investigate CG performance across concept cardinalities (2, 4, 8 concepts) - Version 2 with proper scaling',
            'results': results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Experiment complete!")
    print(f"Results saved to: {output_path}")
    
    # Print summary
    print("\nSummary:")
    print("-" * 40)
    for n_concepts, result in results.items():
        improvement = result['improvement_pct']
        wins = result['cg_wins']
        print(f"{n_concepts} concepts: {improvement:+.2f}% improvement, CG wins: {wins}")
    
    # Analyze pattern
    print("\nAnalysis:")
    print("-" * 40)
    improvements = [results[n]['improvement_pct'] for n in [2, 4, 8]]
    print(f"Improvement pattern: 2 concepts: {improvements[0]:+.2f}%, "
          f"4 concepts: {improvements[1]:+.2f}%, "
          f"8 concepts: {improvements[2]:+.2f}%")
    
    # Determine optimal cardinality
    best_idx = np.argmax(improvements)
    best_n_concepts = [2, 4, 8][best_idx]
    
    if improvements[best_idx] > 0:
        print(f"\nConclusion: CG performs best at {best_n_concepts} concepts with {improvements[best_idx]:+.2f}% improvement")
        if best_n_concepts == 4:
            print("This supports the hypothesis that CG has an optimal complexity at moderate cardinality (4 concepts)")
        elif best_n_concepts == 2:
            print("This suggests CG is actually better for simpler tasks (2 concepts)")
        elif best_n_concepts == 8:
            print("This suggests CG scales well with complexity (8 concepts)")
    else:
        print(f"\nConclusion: CG underperforms baseline at all cardinalities")
        print("This suggests fundamental architectural issues with CG")

if __name__ == '__main__':
    main()