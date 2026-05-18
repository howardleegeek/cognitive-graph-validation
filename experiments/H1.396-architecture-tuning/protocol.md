# H1.396: Architecture Tuning Investigation

## Hypothesis Statement

The CG architecture underperforms baseline in synthetic data due to suboptimal architecture configuration. Adjusting key parameters (hidden dimensions, attention heads, training epochs) will improve CG performance.

## Background

H1.395 showed:
- CG only wins at complexity=100 (+0.7%)
- CG loses at all other complexity levels (avg -4.5%)
- Negative correlation (-0.55) between complexity and CG advantage

Possible causes:
1. **Over-parameterization**: 512-dim unified space may be too large for synthetic data
2. **Attention inefficiency**: Cross-modal attention may add noise for simple patterns
3. **Training insufficiency**: 20 epochs may not be enough for convergence
4. **Learning rate mismatch**: Default LR may not suit this architecture

## Experimental Design

### Test Configurations

| Config | Hidden Dim | Attention Heads | Epochs | LR | Rationale |
|--------|-----------|-----------------|--------|-----|-----------|
| A | 256 | 2 | 20 | 1e-3 | Smaller model |
| B | 512 | 1 | 20 | 1e-3 | Fewer attention heads |
| C | 512 | 4 | 40 | 1e-3 | More training |
| D | 512 | 4 | 20 | 1e-4 | Lower learning rate |
| E | 128 | 1 | 20 | 1e-3 | Minimal model |

### Test Complexity Levels

Focus on complexity=100 (where CG showed slight advantage) and complexity=300 (where CG showed -8% disadvantage).

### Evaluation

- MSE for each configuration
- Compare against baseline (concatenation)
- Identify which architecture changes improve performance

## Predictions

1. Smaller hidden dimensions (Config A, E) will improve performance on simple patterns
2. More training (Config C) will help at higher complexity
3. Lower LR (Config D) may help convergence

## Status

RUNNING