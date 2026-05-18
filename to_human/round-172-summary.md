# Round 172 Summary: Training Dynamics Investigation

**Experiment**: H1.403 - Tested whether CG needs more epochs or different learning rates to show its advantage.

**Key Finding**: CG wins at short training (30 epochs, +11.78% improvement) but loses at longer training (100 epochs, -31.24%). This suggests **overfitting** — CG's additional parameters (attention + GNN) cause it to overfit on small datasets (300 samples), while the simpler baseline generalizes better with longer training.

**Results Summary**:
- CG wins in 2/6 configurations (33% win rate)
- Best configuration: 30 epochs, lr=1e-3 → +11.78% improvement
- Worst configuration: 50 epochs, lr=5e-3 → -47.19% improvement
- Higher learning rates (5e-3) consistently hurt CG (-43.82% avg improvement)
- Longer training (100 epochs) consistently hurts CG (-31.24% avg improvement)

**Implication**: CG's architectural complexity is a double-edged sword. It can capture cross-modal patterns but is prone to overfitting. Next step: test with regularization (dropout, weight decay) to prevent overfitting at longer training epochs.

**Files**: `experiments/H1.403-training-dynamics.py`, `experiments/H1.403-results.json`, `experiments/H1.403-loss-curves.png`