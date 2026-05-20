# Round 235 Summary - H1.469 Multi-Step Tasks Experiment

**Date**: 2026-05-19  
**Experiment**: H1.469 - Test CG on multi-step tasks (3+ steps)  
**Status**: REFUTED

## Key Finding

Cognitive Graph advantage does **NOT** increase with task complexity. Contrary to the hypothesis, CG's improvement over baseline actually **decreases** when moving from single-step to multi-step tasks:

- **Single-step task**: CG improves by **+8.07%** over baseline
- **3-step task**: CG improves by **+2.08%** over baseline  
- **Difference**: **-5.99%** (worse on multi-step)

## Implications

1. **CG is effective for single-step prediction** but its advantage diminishes for multi-step planning
2. The unified representation space may not scale well to longer-horizon tasks without architectural modifications
3. Baseline (separated architecture) handles multi-step tasks relatively better than CG

## Next Action

Based on the priority order, the next action is **H3 re-test: attention on longer sequences (20+ timesteps)**. This follows the H1 deepening attempt and returns to testing attention mechanisms on more complex sequences where they might show advantages over simple concatenation.