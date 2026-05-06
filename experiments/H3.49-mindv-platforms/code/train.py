#!/usr/bin/env python3
"""H3.49: MIND-V Style Semantic Hub on Different Robot Platforms"""

import numpy as np
import torch
import torch.nn as nn


class SRHModel(nn.Module):
    """Semantic Reasoning Hub - MIND-V style"""
    def __init__(self, state_dim: int, lang_dim: int, hidden: int = 64):
        super().__init__()
        self.srh = nn.Sequential(
            nn.Linear(state_dim + lang_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.bsb = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mvg = nn.Sequential(
            nn.Linear(state_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )
        
    def forward(self, state, language):
        task_emb = self.srh(torch.cat([state, language], dim=-1))
        bsb = self.bsb(task_emb)
        return self.mvg(torch.cat([state, bsb], dim=-1))


class DirectMapping(nn.Module):
    """Baseline: Direct state-language concatenation"""
    def __init__(self, state_dim, lang_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + lang_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )
        
    def forward(self, state, language):
        return self.net(torch.cat([state, language], dim=-1))


def simulate_robot_platform(platform_name, state_dim, action_dim, num_samples=50, task_length=10):
    """Simulate a specific robot platform with unique dynamics"""
    np.random.seed(hash(platform_name) % 10000)
    torch.manual_seed(hash(platform_name) % 10000)
    
    # Generate data with platform-specific characteristics
    S = torch.randn(num_samples, task_length, state_dim)
    L = torch.randn(num_samples, task_length, 32)
    
    # Different platforms have different transition dynamics
    if platform_name == "panda_arm":
        # 7-DOF arm - smooth continuous motion
        T = S + torch.randn(num_samples, task_length, state_dim) * 0.1
    elif platform_name == "aloha_bimanual":
        # Bimanual - two arms coordinating
        T = S + torch.randn(num_samples, task_length, state_dim) * 0.15
    elif platform_name == "franka_table":
        # Table-top manipulator - constrained workspace
        T = S + torch.randn(num_samples, task_length, state_dim) * 0.08
    elif platform_name == "ur5_industrial":
        # Industrial arm - precise but less compliant
        T = S + torch.randn(num_samples, task_length, state_dim) * 0.12
    elif platform_name == " WidowX_hover":
        # Hover-style arm
        T = S + torch.randn(num_samples, task_length, state_dim) * 0.14
    else:
        T = S + torch.randn(num_samples, task_length, state_dim) * 0.1
    
    # Train SRH model
    srh = SRHModel(state_dim, 32)
    opt = torch.optim.Adam(srh.parameters(), lr=1e-2)
    for _ in range(30):
        for i in range(num_samples):
            opt.zero_grad()
            pred = srh(S[i,0], L[i,0])
            loss = ((pred - T[i,0])**2).mean()
            loss.backward(); opt.step()
    
    srh_loss = sum(((srh(S[i,0], L[i,0]) - T[i,0])**2).mean().item() for i in range(num_samples)) / num_samples
    
    # Train baseline
    direct = DirectMapping(state_dim, 32)
    opt = torch.optim.Adam(direct.parameters(), lr=1e-2)
    for _ in range(30):
        for i in range(num_samples):
            opt.zero_grad()
            pred = direct(S[i,0], L[i,0])
            loss = ((pred - T[i,0])**2).mean()
            loss.backward(); opt.step()
    
    direct_loss = sum(((direct(S[i,0], L[i,0]) - T[i,0])**2).mean().item() for i in range(num_samples)) / num_samples
    
    improvement = (direct_loss - srh_loss) / direct_loss * 100
    return direct_loss, srh_loss, improvement


def run():
    print("H3.49: MIND-V Semantic Hub on Different Robot Platforms")
    print("=" * 60)
    
    platforms = [
        ("panda_arm", 7, 7),
        ("aloha_bimanual", 14, 14),
        ("franka_table", 7, 7),
        ("ur5_industrial", 6, 6),
        ("widowx_hover", 6, 6),
    ]
    
    results = []
    
    for platform, state_dim, action_dim in platforms:
        print(f"\nTesting {platform} (state_dim={state_dim}, action_dim={action_dim})")
        
        direct_loss, srh_loss, improvement = simulate_robot_platform(
            platform, state_dim, action_dim, num_samples=50, task_length=10
        )
        
        print(f"  Direct: {direct_loss:.4f}, SRH: {srh_loss:.4f}, Δ={improvement:+.1f}%")
        results.append({
            'platform': platform,
            'direct': direct_loss,
            'srh': srh_loss,
            'improvement': improvement
        })
    
    # Cross-platform generalization test
    print("\n" + "=" * 60)
    print("Cross-Platform Generalization Test")
    print("=" * 60)
    
    # Train on one platform, test on others
    generalization_results = []
    
    for train_platform, _, _ in platforms:
        print(f"\nTrain on {train_platform}, test on others:")
        
        # Train on source platform
        np.random.seed(hash(train_platform) % 10000)
        torch.manual_seed(hash(train_platform) % 10000)
        
        S_train = torch.randn(50, 10, 16)
        L_train = torch.randn(50, 10, 32)
        T_train = S_train + torch.randn(50, 10, 16) * 0.1
        
        srh = SRHModel(16, 32)
        opt = torch.optim.Adam(srh.parameters(), lr=1e-2)
        for _ in range(30):
            for i in range(50):
                opt.zero_grad()
                pred = srh(S_train[i,0], L_train[i,0])
                loss = ((pred - T_train[i,0])**2).mean()
                loss.backward(); opt.step()
        
        # Test on all platforms
        for test_platform, _, _ in platforms:
            np.random.seed(hash(test_platform) % 10000)
            torch.manual_seed(hash(test_platform) % 10000)
            
            S_test = torch.randn(30, 10, 16)
            L_test = torch.randn(30, 10, 32)
            
            if test_platform == "aloha_bimanual":
                T_test = S_test + torch.randn(30, 10, 16) * 0.15
            elif test_platform == "franka_table":
                T_test = S_test + torch.randn(30, 10, 16) * 0.08
            elif test_platform == "ur5_industrial":
                T_test = S_test + torch.randn(30, 10, 16) * 0.12
            elif test_platform == "widowx_hover":
                T_test = S_test + torch.randn(30, 10, 16) * 0.14
            else:
                T_test = S_test + torch.randn(30, 10, 16) * 0.1
            
            srh_loss = sum(((srh(S_test[i,0], L_test[i,0]) - T_test[i,0])**2).mean().item() for i in range(30)) / 30
            
            direct = DirectMapping(16, 32)
            opt = torch.optim.Adam(direct.parameters(), lr=1e-2)
            for _ in range(30):
                for i in range(30):
                    opt.zero_grad()
                    pred = direct(S_test[i,0], L_test[i,0])
                    loss = ((pred - T_test[i,0])**2).mean()
                    loss.backward(); opt.step()
            
            direct_loss = sum(((direct(S_test[i,0], L_test[i,0]) - T_test[i,0])**2).mean().item() for i in range(30)) / 30
            
            improvement = (direct_loss - srh_loss) / direct_loss * 100
            print(f"  {train_platform} → {test_platform}: Δ={improvement:+.1f}%")
            generalization_results.append({
                'train': train_platform,
                'test': test_platform,
                'improvement': improvement
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    avg_improvement = np.mean([r['improvement'] for r in results])
    avg_generalization = np.mean([r['improvement'] for r in generalization_results])
    
    print(f"\nPlatform-specific improvement: {avg_improvement:+.1f}%")
    print(f"Cross-platform generalization: {avg_generalization:+.1f}%")
    
    status = "SUPPORTED" if avg_improvement > 5 else "REFUTED" if avg_improvement < -5 else "INCONCLUSIVE"
    print(f"\nStatus: {status}")
    
    return results, generalization_results, status


if __name__ == "__main__":
    run()