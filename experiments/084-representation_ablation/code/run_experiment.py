import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import json
import os
from experiment import run_experiment

if __name__ == "__main__":
    config = {
        'n_train': 400,
        'n_val': 100,
        'n_epochs': 60,
        'batch_size': 32,
        'learning_rate': 1e-3
    }
    
    print("Running H1.386 - Representation Size and Attention Depth Ablation")
    print(f"Config: {config}")
    
    results = run_experiment(config)
    
    # Save results
    os.makedirs('../results', exist_ok=True)
    with open('../results/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    print(f"Baseline MSE: {results['baseline_mse']:.6f}")
    print(f"Hierarchical MSE: {results['hierarchical_mse']:.6f} ({results['hierarchical_improvement']:.2f}% improvement)")
    print(f"Best CG MSE: {results['cg_mse']:.6f} ({results['cg_improvement']:.2f}% improvement)")
    
    print("\nCG Variants Performance:")
    print("-"*60)
    
    # Group variants by type
    rep_variants = [v for v in results['cg_variants'] if v['name'].startswith('CG_physical')]
    head_variants = [v for v in results['cg_variants'] if v['name'].startswith('CG_heads')]
    layer_variants = [v for v in results['cg_variants'] if v['name'].startswith('CG_layers')]
    
    print("\nRepresentation Size Ablation:")
    for v in sorted(rep_variants, key=lambda x: x['mse']):
        print(f"  {v['name']:25} MSE: {v['mse']:.6f} ({v['improvement']:+.2f}%)")
    
    print("\nAttention Heads Ablation:")
    for v in sorted(head_variants, key=lambda x: x['mse']):
        print(f"  {v['name']:25} MSE: {v['mse']:.6f} ({v['improvement']:+.2f}%)")
    
    print("\nGNN Layers Ablation:")
    for v in sorted(layer_variants, key=lambda x: x['mse']):
        print(f"  {v['name']:25} MSE: {v['mse']:.6f} ({v['improvement']:+.2f}%)")
    
    # Determine if CG wins
    cg_wins = results['cg_improvement'] > 0
    print("\n" + "="*60)
    print(f"Cognitive Graph {'WINS' if cg_wins else 'LOSES'}: {results['cg_improvement']:+.2f}% improvement")
    print(f"Hierarchical {'WINS' if results['hierarchical_improvement'] > 0 else 'LOSES'}: {results['hierarchical_improvement']:+.2f}% improvement")
    print("="*60)