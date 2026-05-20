#!/usr/bin/env python3
"""
H1.470.1.1.34: Auxiliary Loss Approaches for Multi-Step Tasks
Following REFUTED curriculum learning (H1.470.1.1.33), test whether
auxiliary losses can improve multi-step task performance.

Prior findings:
- Baseline test loss: 0.016030 (best approach)
- Fixed curriculum: -51.47% worse than baseline
- Adaptive curriculum: -4.61% worse than baseline

Hypothesis: Auxiliary losses (sub-goal prediction, temporal consistency)
will improve multi-step task performance by providing additional gradient
signals without staged training that causes forgetting.

Configurations:
1. Baseline: Standard MSE loss on actions
2. Sub-goal prediction: Auxiliary loss predicting intermediate states
3. Temporal consistency: Loss enforcing smooth transitions between steps
4. Combined: Sub-goal + temporal consistency
5. Weighted auxiliary: Auxiliary losses with adaptive weighting
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path


class CognitiveGraphModel(nn.Module):
    def __init__(self, obs_dim=128, action_dim=7, hidden_dim=64, state_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.processor = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.action_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, action_dim))
        self.subgoal_predictor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, state_dim))
        
    def forward(self, x, return_aux=False):
        h = self.encoder(x)
        out, hidden = self.processor(h)
        actions = self.action_decoder(out)
        if return_aux:
            subgoal_pred = self.subgoal_predictor(hidden[-1])
            return actions, subgoal_pred
        return actions


class MultiStepDataset:
    def __init__(self, n_trajectories=300, max_steps=4, obs_dim=128, action_dim=7, state_dim=32, seed=42):
        self.rng = np.random.RandomState(seed)
        self.max_steps = max_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.proj = self.rng.randn(state_dim, obs_dim) * 0.1
        self.trajectories = self._generate_trajectories(n_trajectories)
        
    def _generate_trajectories(self, n):
        trajectories = []
        for i in range(n):
            n_steps = self.rng.randint(1, self.max_steps + 1)
            states, actions, observations = [], [], []
            current_state = self.rng.randn(self.state_dim) * 0.5
            for step in range(n_steps):
                action = self.rng.randn(self.action_dim) * 0.3
                action += 0.1 * current_state[:self.action_dim]
                action_padded = np.zeros(self.state_dim)
                action_padded[:self.action_dim] = action
                next_state = np.tanh(0.8 * current_state + 0.3 * action_padded + 0.1 * self.rng.randn(self.state_dim))
                obs = current_state @ self.proj + self.rng.randn(self.obs_dim) * 0.05
                states.append(current_state.copy())
                actions.append(action.copy())
                observations.append(obs.copy())
                current_state = next_state
            trajectories.append({'states': np.array(states), 'actions': np.array(actions),
                                 'observations': np.array(observations), 'n_steps': n_steps})
        return trajectories
    
    def get_batch(self, batch_size=32):
        batch_obs, batch_actions, batch_states, batch_next_states = [], [], [], []
        for _ in range(batch_size):
            traj = random.choice(self.trajectories)
            step_idx = self.rng.randint(0, traj['n_steps'])
            obs = np.zeros((self.max_steps, self.obs_dim))
            obs[:traj['n_steps']] = traj['observations']
            actions = np.zeros((self.max_steps, self.action_dim))
            actions[:traj['n_steps']] = traj['actions']
            states = np.zeros((self.max_steps, self.state_dim))
            states[:traj['n_steps']] = traj['states']
            next_states = np.zeros((self.max_steps, self.state_dim))
            if step_idx < traj['n_steps'] - 1:
                next_states[step_idx] = traj['states'][step_idx + 1]
            batch_obs.append(obs); batch_actions.append(actions)
            batch_states.append(states); batch_next_states.append(next_states)
        return {'observations': torch.FloatTensor(np.array(batch_obs)),
                'actions': torch.FloatTensor(np.array(batch_actions)),
                'states': torch.FloatTensor(np.array(batch_states)),
                'next_states': torch.FloatTensor(np.array(batch_next_states))}


class Trainer:
    def __init__(self, config_name, model):
        self.config_name = config_name
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
    def compute_loss(self, batch):
        obs, actions = batch['observations'], batch['actions']
        states, next_states = batch['states'], batch['next_states']
        
        if self.config_name in ['subgoal', 'combined', 'weighted']:
            pred_actions, pred_subgoal = self.model(obs, return_aux=True)
        else:
            pred_actions = self.model(obs)
            pred_subgoal = None
        
        action_loss = F.mse_loss(pred_actions, actions)
        subgoal_loss = torch.tensor(0.0)
        consistency_loss = torch.tensor(0.0)
        total_loss = action_loss
        
        if self.config_name == 'subgoal':
            subgoal_loss = F.mse_loss(pred_subgoal, states[:, -1, :])
            total_loss = action_loss + 0.3 * subgoal_loss
        elif self.config_name == 'consistency':
            _, pred_subgoal = self.model(obs, return_aux=True)
            consistency_loss = F.mse_loss(pred_subgoal, next_states[:, -1, :])
            total_loss = action_loss + 0.2 * consistency_loss
        elif self.config_name == 'combined':
            subgoal_loss = F.mse_loss(pred_subgoal, states[:, -1, :])
            _, pred_subgoal2 = self.model(obs, return_aux=True)
            consistency_loss = F.mse_loss(pred_subgoal2, next_states[:, -1, :])
            total_loss = action_loss + 0.2 * subgoal_loss + 0.15 * consistency_loss
        elif self.config_name == 'weighted':
            subgoal_loss = F.mse_loss(pred_subgoal, states[:, -1, :])
            _, pred_subgoal2 = self.model(obs, return_aux=True)
            consistency_loss = F.mse_loss(pred_subgoal2, next_states[:, -1, :])
            alpha = 0.3 / (1.0 + subgoal_loss.detach())
            beta = 0.2 / (1.0 + consistency_loss.detach())
            total_loss = action_loss + alpha * subgoal_loss + beta * consistency_loss
        
        return total_loss, action_loss.item()
    
    def train(self, dataset, n_epochs=10, batch_size=32, batches_per_epoch=15):
        train_losses = []
        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0
            for _ in range(batches_per_epoch):
                batch = dataset.get_batch(batch_size)
                loss, action_l = self.compute_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / batches_per_epoch)
        return {'final_train_loss': train_losses[-1], 'train_losses': train_losses}


def run_experiment():
    print("=" * 60)
    print("H1.470.1.1.34: Auxiliary Loss Approaches for Multi-Step Tasks")
    print("=" * 60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("\n[1/4] Generating dataset...")
    train_ds = MultiStepDataset(n_trajectories=300, max_steps=4, seed=42)
    test_ds = MultiStepDataset(n_trajectories=100, max_steps=4, seed=123)
    
    configs = [
        ('baseline', 'Baseline: Standard MSE loss'),
        ('subgoal', 'Sub-goal prediction auxiliary loss'),
        ('consistency', 'Temporal consistency auxiliary loss'),
        ('combined', 'Combined sub-goal + temporal consistency'),
        ('weighted', 'Weighted auxiliary losses (adaptive)'),
    ]
    
    results = {}
    print("\n[2/4] Training configurations...")
    for config_name, description in configs:
        print(f"  Training: {config_name}...")
        model = CognitiveGraphModel(obs_dim=128, action_dim=7, hidden_dim=64, state_dim=32)
        trainer = Trainer(config_name, model)
        trainer.train(train_ds, n_epochs=10, batch_size=32, batches_per_epoch=15)
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for _ in range(10):
                batch = test_ds.get_batch(32)
                obs, actions = batch['observations'], batch['actions']
                if config_name in ['subgoal', 'combined', 'weighted']:
                    pred_actions, _ = model(obs, return_aux=True)
                else:
                    pred_actions = model(obs)
                test_loss += F.mse_loss(pred_actions, actions).item()
        test_loss /= 10
        results[config_name] = {'description': description, 'test_action_loss': test_loss}
        print(f"    Test loss: {test_loss:.6f}")
    
    print("\n[3/4] Analyzing results...")
    baseline_test_loss = results['baseline']['test_action_loss']
    
    analysis = {}
    for cn, r in results.items():
        improvement = ((baseline_test_loss - r['test_action_loss']) / baseline_test_loss) * 100
        analysis[cn] = {'test_loss': r['test_action_loss'], 'vs_baseline_pct': improvement}
    
    best_config = min(results.keys(), key=lambda k: results[k]['test_action_loss'])
    best_test_loss = results[best_config]['test_action_loss']
    best_improvement = ((baseline_test_loss - best_test_loss) / baseline_test_loss) * 100
    
    if best_improvement > 2.0:
        conclusion = "SUPPORTED"
    elif best_improvement > -2.0:
        conclusion = "INCONCLUSIVE"
    else:
        conclusion = "REFUTED"
    
    print(f"  Best: {best_config}, loss={best_test_loss:.6f}, improvement={best_improvement:+.2f}%")
    print(f"  Conclusion: {conclusion}")
    
    # Per-complexity analysis
    print("\n[4/4] Per-complexity analysis...")
    complexity_results = {}
    for n_steps in [1, 2, 3, 4]:
        train_ds_s = MultiStepDataset(n_trajectories=100, max_steps=n_steps, seed=42)
        test_ds_s = MultiStepDataset(n_trajectories=50, max_steps=n_steps, seed=123)
        complexity_results[n_steps] = {}
        for cn in ['baseline', best_config]:
            model = CognitiveGraphModel(obs_dim=128, action_dim=7, hidden_dim=64, state_dim=32)
            trainer = Trainer(cn, model)
            trainer.train(train_ds_s, n_epochs=8, batch_size=32, batches_per_epoch=10)
            model.eval()
            tl = 0
            with torch.no_grad():
                for _ in range(5):
                    batch = test_ds_s.get_batch(32)
                    obs, actions = batch['observations'], batch['actions']
                    if cn in ['subgoal', 'combined', 'weighted']:
                        pa, _ = model(obs, return_aux=True)
                    else:
                        pa = model(obs)
                    tl += F.mse_loss(pa, actions).item()
            tl /= 5
            complexity_results[n_steps][cn] = tl
    
    output = {
        'experiment_id': 'H1.470.1.1.34',
        'description': 'Auxiliary loss approaches for multi-step tasks',
        'conclusion': conclusion,
        'task': 'multi_step_manipulation',
        'configurations_tested': len(configs),
        'key_metrics': {
            'baseline_test_loss': round(baseline_test_loss, 6),
            'best_config': best_config,
            'best_test_loss': round(best_test_loss, 6),
            'best_improvement': round(best_improvement, 2),
            'prior_curriculum_best': 0.016030,
            'auxiliary_vs_prior_curriculum': round(((0.016030 - best_test_loss) / 0.016030) * 100, 2),
        },
        'per_config_results': {
            k: {'test_loss': round(v['test_action_loss'], 6), 'vs_baseline_pct': round(analysis[k]['vs_baseline_pct'], 2), 'description': v['description']}
            for k, v in results.items()
        },
        'per_complexity_results': {
            str(k): {'baseline': round(v['baseline'], 6), 'best_aux': round(v[best_config], 6),
                     'improvement_pct': round(((v['baseline'] - v[best_config]) / v['baseline']) * 100, 2)}
            for k, v in complexity_results.items()
        },
        'key_insights': [
            f"Auxiliary losses {'improve' if best_improvement > 0 else 'do not improve'} multi-step task performance by {abs(best_improvement):.2f}%",
            f"Best auxiliary approach: {best_config} (test loss: {best_test_loss:.6f})",
            f"Baseline test loss: {baseline_test_loss:.6f}",
            f"Compared to prior curriculum best (0.016030): {((0.016030 - best_test_loss) / 0.016030) * 100:+.2f}%",
        ],
        'recommendations': [],
        'timestamp': datetime.now().isoformat(),
    }
    
    if conclusion == "SUPPORTED":
        output['recommendations'].append(f"R1: Adopt {best_config} as default for multi-step tasks")
        output['recommendations'].append("R2: Test auxiliary losses on longer sequences (10+ steps)")
        output['recommendations'].append("R3: Investigate optimal auxiliary loss weighting")
    elif conclusion == "INCONCLUSIVE":
        output['recommendations'].append("R1: Test with larger models and more training data")
        output['recommendations'].append("R2: Explore different auxiliary loss formulations")
        output['recommendations'].append("R3: Consider combining auxiliary losses with regularization")
    else:
        output['recommendations'].append("R1: Auxiliary losses do not help; baseline remains best")
        output['recommendations'].append("R2: Investigate architectural changes instead of loss modifications")
        output['recommendations'].append("R3: Consider whether multi-step tasks need fundamentally different approach")
    
    results_path = Path(__file__).parent / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"Experiment complete. Conclusion: {conclusion}")
    return output


if __name__ == '__main__':
    run_experiment()
