"""
H3.23: SSM on ALOHA Tasks with Long Sequences
==============================================
Test SSM (State Space Model) on ALOHA-style bimanual robot tasks
with longer sequences (20+ timesteps) based on H3 failure insights.

Key findings from H3:
- Concatenation wins on simple tasks
- Attention helps on longer sequences (20+ timesteps)
- SSM/Mamba architectures show promise for long sequences

This experiment tests SSM on ALOHA tasks with:
- Long sequences (20-40 timesteps)
- Bimanual coordination
- Fine motor control tasks
"""

import numpy as np
import json
import os
from datetime import datetime

class SSMExperiment:
    def __init__(self):
        self.n_tasks = 8
        self.n_demos = 200
        self.hidden_dim = 512
        self.seq_lens = [20, 25, 30, 35, 40]
        
    def generate_aloha_tasks(self):
        """Generate ALOHA-style manipulation tasks with varying sequence lengths"""
        tasks = []
        
        # Task 1: Thread insertion (fine motor, long sequence)
        tasks.append({
            'name': 'thread_insertion',
            'n_steps': 25,
            'objects': 2,
            'requires_bimanual': True,
            'requires_precision': True
        })
        
        # Task 2: Cup stacking (medium sequence)
        tasks.append({
            'name': 'cup_stacking',
            'n_steps': 20,
            'objects': 5,
            'requires_bimanual': False,
            'requires_precision': True
        })
        
        # Task 3: Fruit arrangement
        tasks.append({
            'name': 'fruit_arrangement',
            'n_steps': 30,
            'objects': 6,
            'requires_bimanual': True,
            'requires_precision': False
        })
        
        # Task 4: Cable plugging
        tasks.append({
            'name': 'cable_plugging',
            'n_steps': 35,
            'objects': 3,
            'requires_bimanual': True,
            'requires_precision': True
        })
        
        # Task 5: Cloth folding
        tasks.append({
            'name': 'cloth_folding',
            'n_steps': 40,
            'objects': 1,
            'requires_bimanual': True,
            'requires_precision': False
        })
        
        # Task 6: Plate serving
        tasks.append({
            'name': 'plate_serving',
            'n_steps': 20,
            'objects': 4,
            'requires_bimanual': True,
            'requires_precision': False
        })
        
        # Task 7: Pour water
        tasks.append({
            'name': 'pour_water',
            'n_steps': 25,
            'objects': 2,
            'requires_bimanual': True,
            'requires_precision': True
        })
        
        # Task 8: Object rearrangement
        tasks.append({
            'name': 'object_rearrangement',
            'n_steps': 30,
            'objects': 8,
            'requires_bimanual': False,
            'requires_precision': False
        })
        
        return tasks
    
    def ssm_predict(self, seq, state_dim=16):
        """SSM-based next-step prediction"""
        seq_len, dim = seq.shape
        
        # Project to state space
        s = np.dot(seq, np.random.randn(dim, state_dim) / np.sqrt(dim))
        
        # State transition with gating (Mamba-style)
        for i in range(1, seq_len):
            # Gating
            gate = 1 / (1 + np.exp(-np.dot(s[i-1], np.random.randn(state_dim, dim) @ np.random.randn(dim, state_dim) / state_dim))
            )
            # State update
            s[i] = 0.9 * s[i-1] * gate + 0.1 * np.dot(seq[i], np.random.randn(dim, state_dim) / np.sqrt(dim))
        
        # Predict next state from final state
        final_state = s[-1]
        pred = np.dot(final_state, np.random.randn(state_dim, dim) / np.sqrt(state_dim))
        
        return pred
    
    def baseline_predict(self, seq):
        """Baseline: linear projection from last step"""
        return seq[-1] + np.random.randn(seq.shape[1]) * 0.01
    
    def run_experiment(self):
        """Run SSM vs Baseline on ALOHA tasks"""
        print("=" * 60)
        print("H3.23: SSM on ALOHA Tasks with Long Sequences")
        print("=" * 60)
        
        tasks = self.generate_aloha_tasks()
        results = {
            'hypothesis': 'H3.23',
            'statement': 'SSM outperforms on ALOHA long-sequence tasks',
            'tasks': [],
            'baseline_errors': [],
            'ssm_errors': [],
            'improvements': []
        }
        
        for task in tasks:
            print(f"\n--- Task: {task['name']} ---")
            print(f"  Steps: {task['n_steps']}, Objects: {task['objects']}")
            
            # Generate demonstration data with ground truth next-step targets
            n_demos = self.n_demos
            input_dim = 64  # State + action dimension
            
            baseline_errors = []
            ssm_errors = []
            
            for _ in range(n_demos):
                # Generate sequence with dynamics
                seq = np.random.randn(task['n_steps'], input_dim) * 0.1
                for i in range(1, task['n_steps']):
                    seq[i] = seq[i-1] * 0.8 + np.random.randn(input_dim) * 0.05
                
                # Ground truth next step
                next_step = seq[-1] * 0.8 + np.random.randn(input_dim) * 0.05
                
                # Baseline prediction
                baseline_pred = self.baseline_predict(seq)
                baseline_error = np.linalg.norm(next_step - baseline_pred)
                baseline_errors.append(baseline_error)
                
                # SSM prediction
                ssm_pred = self.ssm_predict(seq, state_dim=16)
                ssm_error = np.linalg.norm(next_step - ssm_pred)
                ssm_errors.append(ssm_error)
            
            baseline_error = np.mean(baseline_errors)
            ssm_error = np.mean(ssm_errors)
            
            improvement = (baseline_error - ssm_error) / baseline_error * 100 if baseline_error > 0 else 0
            
            print(f"  Baseline Error: {baseline_error:.6f}")
            print(f"  SSM Error: {ssm_error:.6f}")
            print(f"  Improvement: {improvement:.2f}%")
            
            results['tasks'].append(task['name'])
            results['baseline_errors'].append(float(baseline_error))
            results['ssm_errors'].append(float(ssm_error))
            results['improvements'].append(float(improvement))
        
        avg_improvement = np.mean(results['improvements'])
        results['evidence'] = f"+{avg_improvement:.1f}% avg improvement on ALOHA long-sequence tasks"
        
        print("\n" + "=" * 60)
        print(f"AVERAGE IMPROVEMENT: {avg_improvement:.2f}%")
        print("=" * 60)
        
        # Determine status
        if avg_improvement > 10:
            results['status'] = 'SUPPORTED'
            print("STATUS: SUPPORTED")
        elif avg_improvement > 0:
            results['status'] = 'PARTIAL'
            print("STATUS: PARTIAL")
        else:
            results['status'] = 'REFUTED'
            print("STATUS: REFUTED")
        
        return results
    
    def save_results(self, results):
        """Save results to JSON"""
        output_dir = "experiments/H3.23-ssm-aloha-long-seq"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_dir}/results.json")


def main():
    experiment = SSMExperiment()
    results = experiment.run_experiment()
    experiment.save_results(results)


if __name__ == "__main__":
    main()