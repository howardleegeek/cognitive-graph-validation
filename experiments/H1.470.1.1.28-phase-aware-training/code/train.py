#!/usr/bin/env python3
"""
H1.470.1.1.28: Phase-Aware Training for Hierarchical Tasks (Fast Version)

Context:
- H1.470.1.1.27 showed ensemble disagreement fails on hierarchical tasks (-1.35% to -1.59%)
- The key insight: phase transitions are "uncertain" but highly informative
- Ensemble disagreement downweights phase transitions, removing learning signal

Hypothesis: Phase-aware training that UPWEIGHTS phase transition samples will
improve performance on hierarchical multi-step tasks, unlike noise-aware loss
that downweights them.

Test configurations:
1. Baseline: Standard training without phase weighting
2. Oracle phase: Ground truth phase labels (upper bound)
3. Detected phase: Automatic phase transition detection
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from pathlib import Path


class HierarchicalPhaseDataset(Dataset):
    """Generate hierarchical manipulation trajectories with phase transitions."""
    
    def __init__(self, n_samples, seq_len, n_phases=3, seed=None):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_phases = n_phases
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        torch.manual_seed(idx)
        np.random.seed(idx)
        
        # Generate phase boundaries
        phase_lengths = np.random.randint(
            self.seq_len // (self.n_phases + 2),
            self.seq_len // self.n_phases + 2,
            size=self.n_phases
        )
        phase_lengths = np.maximum(phase_lengths, 3)  # Min 3 steps per phase
        total = phase_lengths.sum()
        if total > self.seq_len:
            phase_lengths = phase_lengths * self.seq_len // total
        phase_lengths = phase_lengths.astype(int)
        
        # Pad to seq_len
        while phase_lengths.sum() < self.seq_len:
            phase_lengths[np.random.randint(self.n_phases)] += 1
        
        # Generate trajectory with phase transitions
        positions = torch.zeros(self.seq_len, 4)
        phase_labels = torch.zeros(self.seq_len, dtype=torch.long)
        phase_transitions = torch.zeros(self.seq_len)
        
        current_idx = 0
        
        for phase_id in range(self.n_phases):
            phase_len = phase_lengths[phase_id] if phase_id < len(phase_lengths) else 5
            phase_len = min(phase_len, self.seq_len - current_idx)
            if phase_len <= 0:
                break
                
            # Each phase has a distinct target region
            target = torch.randn(4) * 2.0
            
            # Generate smooth trajectory toward target
            for i in range(phase_len):
                if current_idx + i >= self.seq_len:
                    break
                progress = i / max(phase_len - 1, 1)
                noise = torch.randn(4) * 0.1
                positions[current_idx + i] = target * progress + noise
                phase_labels[current_idx + i] = phase_id
            
            # Mark phase transition
            if phase_id > 0 and current_idx < self.seq_len:
                phase_transitions[current_idx] = 1.0
            
            current_idx += phase_len
        
        # Fill remaining with last phase
        while current_idx < self.seq_len:
            positions[current_idx] = positions[current_idx - 1] + torch.randn(4) * 0.05
            phase_labels[current_idx] = self.n_phases - 1
            current_idx += 1
        
        # Compute actions
        actions = torch.zeros(self.seq_len, 4)
        actions[:-1] = positions[1:] - positions[:-1]
        
        # Language instruction (one-hot task type)
        lang = torch.zeros(16)
        lang[idx % 16] = 1.0
        
        return {
            'observation': positions,
            'action': actions,
            'language': lang,
            'phase_labels': phase_labels,
            'phase_transitions': phase_transitions,
            'n_phases': self.n_phases
        }


class BaselineCognitiveGraph(nn.Module):
    """Baseline Cognitive Graph without phase awareness."""
    
    def __init__(self, obs_dim=4, action_dim=4, lang_dim=16, hidden=64):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, hidden)
        self.lang_encoder = nn.Linear(lang_dim, hidden)
        self.graph = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden // 2, action_dim)
    
    def forward(self, obs, lang):
        obs_emb = self.obs_encoder(obs[:, -1])  # Last observation
        lang_emb = self.lang_encoder(lang)
        combined = torch.cat([obs_emb, lang_emb], dim=-1)
        graph_out = self.graph(combined)
        return self.action_head(graph_out)


class PhaseAwareCognitiveGraph(nn.Module):
    """Cognitive Graph with phase-aware attention."""
    
    def __init__(self, obs_dim=4, action_dim=4, lang_dim=16, hidden=64, n_phases=5):
        super().__init__()
        self.n_phases = n_phases
        self.obs_encoder = nn.Linear(obs_dim, hidden)
        self.lang_encoder = nn.Linear(lang_dim, hidden)
        self.phase_encoder = nn.Embedding(n_phases, hidden // 4)
        
        # Output projection
        self.output_proj = nn.Linear(hidden + hidden // 4, hidden)
        
        self.action_head = nn.Linear(hidden, action_dim)
    
    def forward(self, obs, lang, phase_labels=None):
        batch_size, seq_len, _ = obs.shape
        
        # Encode all observations in sequence
        obs_emb = self.obs_encoder(obs[:, -1])  # Last observation
        lang_emb = self.lang_encoder(lang)
        combined = obs_emb + lang_emb
        
        if phase_labels is not None:
            # Clamp phase labels to valid range
            phase_labels = torch.clamp(phase_labels, 0, self.n_phases - 1)
            phase_emb = self.phase_encoder(phase_labels[:, -1])  # [B, H//4]
            # Combine with phase info
            combined = torch.cat([combined, phase_emb], dim=-1)
        else:
            phase_emb = torch.zeros(batch_size, hidden // 4, device=obs.device)
            combined = torch.cat([combined, phase_emb], dim=-1)
        
        output = self.output_proj(combined)
        return self.action_head(output)


class OraclePhaseAwareCognitiveGraph(nn.Module):
    """Phase-aware model with oracle phase transition detection."""
    
    def __init__(self, obs_dim=4, action_dim=4, lang_dim=16, hidden=64, n_phases=5):
        super().__init__()
        self.n_phases = n_phases
        self.obs_encoder = nn.Linear(obs_dim, hidden)
        self.lang_encoder = nn.Linear(lang_dim, hidden)
        self.phase_encoder = nn.Embedding(n_phases, hidden // 4)
        self.transition_encoder = nn.Linear(1, hidden // 4)
        
        # Output projection
        self.output_proj = nn.Linear(hidden + hidden // 2, hidden)
        
        self.action_head = nn.Linear(hidden, action_dim)
    
    def forward(self, obs, lang, phase_labels=None, phase_transitions=None):
        batch_size, seq_len, _ = obs.shape
        
        obs_emb = self.obs_encoder(obs[:, -1])  # Last observation
        lang_emb = self.lang_encoder(lang)
        combined = obs_emb + lang_emb
        
        if phase_labels is not None and phase_transitions is not None:
            # Clamp phase labels to valid range
            phase_labels = torch.clamp(phase_labels, 0, self.n_phases - 1)
            phase_emb = self.phase_encoder(phase_labels[:, -1])  # [B, H//4]
            trans_emb = self.transition_encoder(phase_transitions[:, -1].unsqueeze(-1))  # [B, H//4]
            phase_info = torch.cat([phase_emb, trans_emb], dim=-1)  # [B, H//2]
            combined = torch.cat([combined, phase_info], dim=-1)
        else:
            phase_info = torch.zeros(batch_size, hidden // 2, device=obs.device)
            combined = torch.cat([combined, phase_info], dim=-1)
        
        output = self.output_proj(combined)
        return self.action_head(output)


def detect_phase_transitions(obs, threshold=0.3):
    """Automatically detect phase transitions from observation changes."""
    batch_size, seq_len, obs_dim = obs.shape
    
    transitions = torch.zeros(batch_size, seq_len, device=obs.device)
    
    if seq_len > 2:
        velocities = obs[:, 1:] - obs[:, :-1]  # [B, T-1, D]
        vel_changes = velocities[:, 1:] - velocities[:, :-1]  # [B, T-2, D]
        change_mags = torch.norm(vel_changes, dim=-1)  # [B, T-2]
        
        # Normalize
        change_mags = change_mags / (change_mags.max(dim=1, keepdim=True)[0] + 1e-6)
        
        transitions[:, 2:] = (change_mags > threshold).float()
    
    return transitions


def infer_phase_labels(transitions, n_phases=5):
    """Infer phase labels from detected transitions."""
    batch_size, seq_len = transitions.shape
    phase_labels = torch.zeros(batch_size, seq_len, dtype=torch.long, device=transitions.device)
    
    for b in range(batch_size):
        phase = 0
        for t in range(seq_len):
            if transitions[b, t] > 0.5 and phase < n_phases - 1:
                phase += 1
            phase_labels[b, t] = phase
    
    return phase_labels


def train_epoch(model, dataloader, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_samples = 0
    
    phase_weight = config.get('phase_weight', 2.0)
    use_oracle_phase = config.get('oracle_phase', False)
    use_detected_phase = config.get('detected_phase', False)
    
    for batch in dataloader:
        obs = batch['observation'].to(device)
        actions = batch['action'].to(device)
        lang = batch['language'].to(device)
        phase_labels = batch['phase_labels'].to(device)
        phase_transitions = batch['phase_transitions'].to(device)
        
        optimizer.zero_grad()
        
        if use_oracle_phase:
            pred_actions = model(obs, lang, phase_labels, phase_transitions)
        elif use_detected_phase:
            detected_trans = detect_phase_transitions(obs)
            detected_labels = infer_phase_labels(detected_trans, n_phases=config['n_phases'] + 1)
            pred_actions = model(obs, lang, detected_labels)
        else:
            pred_actions = model(obs, lang)
        
        # Standard loss
        loss = F.mse_loss(pred_actions, actions[:, -1])
        
        # Phase-weighted loss (upweight transitions)
        if use_oracle_phase or use_detected_phase:
            transition_mask = phase_transitions
            sample_weight = 1.0 + (phase_weight - 1.0) * transition_mask[:, -1]
            weighted_loss = (loss * sample_weight).mean()
        else:
            weighted_loss = loss
        
        weighted_loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * obs.size(0)
        n_samples += obs.size(0)
    
    return total_loss / n_samples


def evaluate(model, dataloader, device, config):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    n_samples = 0
    
    use_oracle_phase = config.get('oracle_phase', False)
    use_detected_phase = config.get('detected_phase', False)
    
    with torch.no_grad():
        for batch in dataloader:
            obs = batch['observation'].to(device)
            actions = batch['action'].to(device)
            lang = batch['language'].to(device)
            phase_labels = batch['phase_labels'].to(device)
            phase_transitions = batch['phase_transitions'].to(device)
            
            if use_oracle_phase:
                pred_actions = model(obs, lang, phase_labels, phase_transitions)
            elif use_detected_phase:
                detected_trans = detect_phase_transitions(obs)
                detected_labels = infer_phase_labels(detected_trans, n_phases=config['n_phases'] + 1)
                pred_actions = model(obs, lang, detected_labels)
            else:
                pred_actions = model(obs, lang)
            
            loss = F.mse_loss(pred_actions, actions[:, -1])
            total_loss += loss.item() * obs.size(0)
            n_samples += obs.size(0)
    
    return total_loss / n_samples


def run_experiment(config):
    """Run single experiment configuration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = HierarchicalPhaseDataset(
        n_samples=config['n_train'],
        seq_len=config['seq_len'],
        n_phases=config['n_phases'],
        seed=42
    )
    test_dataset = HierarchicalPhaseDataset(
        n_samples=config['n_test'],
        seq_len=config['seq_len'],
        n_phases=config['n_phases'],
        seed=123
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    if config.get('oracle_phase', False):
        model = OraclePhaseAwareCognitiveGraph(
            hidden=config['hidden'],
            n_phases=config['n_phases'] + 1
        ).to(device)
    elif config.get('detected_phase', False):
        model = PhaseAwareCognitiveGraph(
            hidden=config['hidden'],
            n_phases=config['n_phases'] + 1
        ).to(device)
    else:
        model = BaselineCognitiveGraph(hidden=config['hidden']).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # Training loop
    results = {'test_loss': []}
    
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, device, config)
        test_loss = evaluate(model, test_loader, device, config)
        results['test_loss'].append(test_loss)
        
        if epoch % 5 == 0 or epoch == config['epochs'] - 1:
            print(f"Epoch {epoch}: train={train_loss:.6f}, test={test_loss:.6f}")
    
    return results


def main():
    print("=" * 60)
    print("H1.470.1.1.28: Phase-Aware Training for Hierarchical Tasks")
    print("=" * 60)
    
    base_config = {
        'n_train': 500,
        'n_test': 100,
        'batch_size': 64,
        'hidden': 64,
        'epochs': 30,
        'lr': 1e-3,
    }
    
    all_results = {}
    
    # Test different phase configurations
    for n_phases in [3, 4, 5]:
        print(f"\n{'='*60}")
        print(f"Testing with {n_phases} phases")
        print(f"{'='*60}")
        
        config = {**base_config, 'n_phases': n_phases, 'seq_len': n_phases * 10}
        
        # 1. Baseline
        print(f"\n[1/3] Baseline (no phase awareness)")
        baseline_config = {**config, 'oracle_phase': False, 'detected_phase': False}
        baseline_results = run_experiment(baseline_config)
        
        # 2. Oracle Phase (upper bound)
        print(f"\n[2/3] Oracle Phase (ground truth phases)")
        oracle_config = {**config, 'oracle_phase': True, 'detected_phase': False, 'phase_weight': 2.0}
        oracle_results = run_experiment(oracle_config)
        
        # 3. Detected Phase (automatic)
        print(f"\n[3/3] Detected Phase (automatic detection)")
        detected_config = {**config, 'oracle_phase': False, 'detected_phase': True, 'phase_weight': 2.0}
        detected_results = run_experiment(detected_config)
        
        # Store results
        baseline_loss = baseline_results['test_loss'][-1]
        oracle_loss = oracle_results['test_loss'][-1]
        detected_loss = detected_results['test_loss'][-1]
        
        oracle_improvement = (baseline_loss - oracle_loss) / baseline_loss * 100
        detected_improvement = (baseline_loss - detected_loss) / baseline_loss * 100
        
        all_results[f'{n_phases}_phase'] = {
            'baseline_test_loss': baseline_loss,
            'oracle_test_loss': oracle_loss,
            'detected_test_loss': detected_loss,
            'oracle_improvement': oracle_improvement,
            'detected_improvement': detected_improvement
        }
        
        print(f"\n{n_phases}-Phase Results:")
        print(f"  Baseline test loss: {baseline_loss:.6f}")
        print(f"  Oracle phase loss: {oracle_loss:.6f} ({oracle_improvement:+.2f}%)")
        print(f"  Detected phase loss: {detected_loss:.6f} ({detected_improvement:+.2f}%)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    summary = {
        'experiment_id': 'H1.470.1.1.28',
        'timestamp': str(np.datetime64('now')),
        'configurations_tested': 3,
        'architectures_tested': ['baseline', 'oracle_phase', 'detected_phase'],
        'phase_counts': [3, 4, 5],
        'results': all_results,
        'conclusion': None
    }
    
    # Determine conclusion
    total_oracle_improvement = 0
    total_detected_improvement = 0
    n_configs = 0
    
    for phase_key in ['3_phase', '4_phase', '5_phase']:
        if phase_key in all_results:
            total_oracle_improvement += all_results[phase_key]['oracle_improvement']
            total_detected_improvement += all_results[phase_key]['detected_improvement']
            n_configs += 1
    
    avg_oracle = total_oracle_improvement / n_configs if n_configs > 0 else 0
    avg_detected = total_detected_improvement / n_configs if n_configs > 0 else 0
    
    if avg_oracle > 5.0 and avg_detected > 2.0:
        summary['conclusion'] = 'SUPPORTED'
        summary['key_insight'] = f'Phase-aware training improves hierarchical task learning (oracle: {avg_oracle:.2f}%, detected: {avg_detected:.2f}%)'
    elif avg_oracle > 2.0:
        summary['conclusion'] = 'PARTIALLY_SUPPORTED'
        summary['key_insight'] = f'Oracle phase labels help ({avg_oracle:.2f}%) but automatic detection needs work ({avg_detected:.2f}%)'
    else:
        summary['conclusion'] = 'REFUTED'
        summary['key_insight'] = f'Phase-aware training does not help hierarchical tasks (oracle: {avg_oracle:.2f}%, detected: {avg_detected:.2f}%)'
    
    summary['avg_oracle_improvement'] = avg_oracle
    summary['avg_detected_improvement'] = avg_detected
    
    print(f"\nAverage Oracle Improvement: {avg_oracle:.2f}%")
    print(f"Average Detected Improvement: {avg_detected:.2f}%")
    print(f"Conclusion: {summary['conclusion']}")
    
    # Save results
    results_path = Path(__file__).parent.parent / 'results' / 'metrics.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return summary


if __name__ == '__main__':
    main()