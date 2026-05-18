# Round 190 Summary: Hybrid Cognitive Graph Architecture Test

## Experiment Executed
**H1.424**: Designed and tested a hybrid Cognitive Graph architecture that adaptively selects between Per-Object CG (for short horizons ≤20 steps) and 2-Node CG (for long horizons >20 steps) based on sequence length.

## Key Results
- **Performance**: Hybrid architecture underperformed baseline by -8.82% (MSE: 3.217 vs 2.956)
- **Selection Failure**: Selector preferred two-node architecture (62.2% weight) even at seq_len=15 where per-object should be advantageous
- **High Variance**: Selection weights showed extreme variation across samples (0.06 to 0.94 per-object weight)
- **Training Issues**: Hybrid model showed worse generalization (val loss 3.210 vs baseline 2.896)

## Key Insight
Naive adaptive architecture selection fails without proper guidance. Simply providing sequence length as input to a selector network is insufficient for learning optimal architecture choices. The selector needs auxiliary supervision, curriculum training across sequence lengths, or reinforcement learning to properly learn when to use each architecture.

## Next Step
**H1.425**: Improve hybrid architecture training with auxiliary loss to guide selector towards correct architecture choice based on sequence length, and test with curriculum learning across multiple sequence lengths.

## Files Created/Modified
- `experiments/084-hybrid_architecture/code/experiment.py` - Main experiment code
- `experiments/084-hybrid_architecture/code/experiment_v2.py` - Simplified version with synthetic data
- `experiments/084-hybrid_architecture/code/test_hybrid.py` - Test script
- `experiments/084-hybrid_architecture/code/test_simple.py` - Basic functionality tests
- `experiments/084-hybrid_architecture/results/results_20260518_071056.json` - Experiment results
- `research-state.yaml` - Updated with round 190 results
- `findings.md` - Updated with H1.424 findings
- `to_human/round-190-summary.md` - This summary