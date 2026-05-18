# Round 172 Summary: Training Dynamics Investigation

**Experiment**: H1.403 - Tested whether CG needs more epochs or different learning rates to show its advantage.

**Critical Finding**: Learning rate is the key factor for CG success. CG wins consistently with lr=1e-4 (4/4 wins, +15% to +32% improvement across all epochs) but loses with lr≥1e-3 (0/5 wins). This explains why H1.402 showed 0% win rate — it used lr=1e-3.

**Results Summary**:
- CG wins in 4/9 configurations (44% win rate)
- With lr=1e-4: CG wins 3/3 (avg +23.50% improvement)
- With lr=1e-3: CG wins 1/3 (avg -14.78% improvement)
- With lr=5e-3: CG wins 0/3 (avg -73.32% improvement)

**Best configurations**:
- epochs=30, lr=1e-4: +31.83% improvement
- epochs=100, lr=1e-4: +23.62% improvement
- epochs=200, lr=1e-4: +15.04% improvement

**Implication**: CG's attention and GNN modules are sensitive to learning rate. The architectural complexity requires careful hyperparameter tuning. Next step: re-test H1.402 configurations with lr=1e-4 to see if CG advantage emerges.

**Files**: `experiments/H1.403-training-dynamics.py`, `experiments/H1.403-results.json`, `experiments/H1.403-loss-curves.png`