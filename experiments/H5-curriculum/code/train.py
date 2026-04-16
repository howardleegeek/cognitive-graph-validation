"""
H5: Curriculum Learning Experiment
Tests if pre-training physical, then adding semantic outperforms joint training.
Based on curriculum learning hypothesis from DeepMind work.
"""

import sys
sys.path.insert(0, "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json


class SimpleManipDataset(Dataset):
    def __init__(self, n_samples: int = 500, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.objects = torch.randn(n_samples, 8)
        self.objects[:, 4:7] = torch.softmax(self.objects[:, 4:7], dim=1)
        self.instructions = torch.randn(n_samples, 32)
        self.actions = torch.randn(n_samples, 5)
        self.actions[:, 3:5] = torch.sigmoid(self.actions[:, 3:5])

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        return {
            "observation": self.objects[idx],
            "language": self.instructions[idx],
            "action": self.actions[idx],
        }


class JointTrainingModel(nn.Module):
    """Standard joint training from scratch."""

    def __init__(self, obs_dim=8, lang_dim=32, action_dim=5, phys_dim=112, sem_dim=400):
        super().__init__()
        total_dim = phys_dim + sem_dim
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, phys_dim), nn.LayerNorm(phys_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, sem_dim), nn.LayerNorm(sem_dim)
        )
        
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim))
            for _ in range(3)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs, lang):
        z_phys = self.obs_encoder(obs)
        z_sem = self.lang_encoder(lang)
        nodes = torch.cat([z_phys, z_sem], dim=-1)
        for layer in self.fusion_layers:
            nodes = nodes + layer(nodes)
        return self.decoder(nodes)


class CurriculumTrainingModel(nn.Module):
    """Pre-train physical, then add semantic."""

    def __init__(self, obs_dim=8, lang_dim=32, action_dim=5, phys_dim=112, sem_dim=400):
        super().__init__()
        total_dim = phys_dim + sem_dim
        self.phys_dim = phys_dim
        self.sem_dim = sem_dim
        
        # Physical encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, phys_dim), nn.LayerNorm(phys_dim)
        )
        
        # Semantic encoder (frozen initially)
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, sem_dim), nn.LayerNorm(sem_dim)
        )
        
        # Fusion layers with curriculum control
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim))
            for _ in range(3)
        ])
        
        # Separate decoders for phase 1 and 2
        self.decoder_phase1 = nn.Sequential(
            nn.Linear(phys_dim, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward_phase1(self, obs):
        """Phase 1: Physical only."""
        z_phys = self.obs_encoder(obs)
        return self.decoder_phase1(z_phys)

    def forward_phase2(self, obs, lang):
        """Phase 2: Full unified."""
        z_phys = self.obs_encoder(obs)
        z_sem = self.lang_encoder(lang)
        nodes = torch.cat([z_phys, z_sem], dim=-1)
        for layer in self.fusion_layers:
            nodes = nodes + layer(nodes)
        return self.decoder(nodes)
    
    def forward(self, obs, lang):
        return self.forward_phase2(obs, lang)


def train_epoch(model, loader, optimizer, criterion, phase=None):
    model.train()
    losses = []
    for batch in loader:
        optimizer.zero_grad()
        
        if phase == 1 and hasattr(model, 'forward_phase1'):
            pred = model.forward_phase1(batch["observation"])
        else:
            pred = model(batch["observation"], batch["language"])
            
        loss = criterion(pred, batch["action"])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def eval_model(model, loader, criterion, phase=None):
    model.eval()
    with torch.no_grad():
        losses = []
        for batch in loader:
            if phase == 1 and hasattr(model, 'forward_phase1'):
                pred = model.forward_phase1(batch["observation"])
            else:
                pred = model(batch["observation"], batch["language"])
            loss = criterion(pred, batch["action"])
            losses.append(loss.item())
    return np.mean(losses)


def run_curriculum_experiment(n_samples=500):
    print("=" * 70)
    print("H5 CURRICULUM LEARNING")
    print("Testing: Pre-train physical then add semantic vs joint training")
    print("=" * 70)

    # Create datasets
    train_data = SimpleManipDataset(n_samples=n_samples, seed=42)
    val_data = SimpleManipDataset(n_samples=200, seed=99)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    
    criterion = nn.MSELoss()
    results = {}
    
    # === JOINT TRAINING (Baseline) ===
    print("\n[Phase 1] Joint Training (baseline)...")
    joint_model = JointTrainingModel()
    optimizer = torch.optim.Adam(joint_model.parameters(), lr=1e-3)
    
    for epoch in range(100):
        train_epoch(joint_model, train_loader, optimizer, criterion)
    
    joint_loss = eval_model(joint_model, val_loader, criterion)
    results["joint"] = joint_loss
    print(f"Joint training final loss: {joint_loss:.4f}")
    
    # === CURRICULUM TRAINING ===
    print("\n[Phase 2] Curriculum Training...")
    curriculum_model = CurriculumTrainingModel()
    optimizer_cur = torch.optim.Adam(curriculum_model.parameters(), lr=1e-3)
    
    # Phase 1: Physical only (epochs 1-30)
    print("  Phase 1: Physical encoder training...")
    for epoch in range(30):
        train_epoch(curriculum_model, train_loader, optimizer_cur, criterion, phase=1)
    
    phase1_loss = eval_model(curriculum_model, val_loader, criterion, phase=1)
    print(f"    Phase 1 val loss: {phase1_loss:.4f}")
    
    # Phase 2: Full unified (epochs 31-100)
    print("  Phase 2: Full unified training...")
    for epoch in range(70):
        train_epoch(curriculum_model, train_loader, optimizer_cur, criterion, phase=2)
    
    curriculum_loss = eval_model(curriculum_model, val_loader, criterion)
    results["curriculum"] = curriculum_loss
    print(f"Curriculum final loss: {curriculum_loss:.4f}")
    
    # === RESULTS ===
    improvement = (joint_loss - curriculum_loss) / joint_loss * 100
    
    print("\n" + "=" * 70)
    print("H5 RESULTS SUMMARY - Curriculum Learning")
    print("=" * 70)
    print(f"Joint Training:    {joint_loss:.4f}")
    print(f"Curriculum:        {curriculum_loss:.4f}")
    print(f"Improvement:       {improvement:+.1f}%")
    print(f"\nH5: {'SUPPORTED' if improvement > 0 else 'REFUTED'}")
    
    # Save metrics
    import os
    os.makedirs("experiments/H5-curriculum/results", exist_ok=True)
    with open("experiments/H5-curriculum/results/h5_metrics.json", "w") as f:
        json.dump({
            "joint_loss": float(joint_loss),
            "curriculum_loss": float(curriculum_loss),
            "improvement_percent": float(improvement),
            "hypothesis_supported": bool(improvement > 0)
        }, f, indent=2)
    
    return results


if __name__ == "__main__":
    run_curriculum_experiment(n_samples=500)