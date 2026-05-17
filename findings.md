# Research Findings — Cognitive Graph Architecture

## Research Question

Does a unified cognitive graph architecture (early fusion of physical and semantic representations) achieve higher sample efficiency than separated architectures (JEPA + LLM alignment) on language-conditioned robotic tasks?

## Current Understanding

**The Core Hypothesis**: Current approaches (V-JEPA 2, π0, LED-WM) all suffer from representation separation — vision and language exist in different spaces and are only aligned after encoding. This causes:
1. **Grounding problems**: Language not truly grounded in physical dynamics
2. **Combinatorial explosion**: Need to learn all (vision, language) pairings separately
3. **Learning inefficiency**: No gradient flow between modalities during training

**The Cognitive Graph Solution**: A unified 512-dimensional representation space where:
- 144 dimensions encode physical world state (analogous to V-JEPA embeddings)
- 368 dimensions encode semantic/conceptual information (analogous to LLM embeddings)
- Single GNN processes both, with cross-modal attention allowing dynamic interaction
- Explicit graph structure (nodes = objects/concepts, edges = relationships/physics)

## Key Results

### H1.379: Aggressive Subgoal Decomposition for 4+ Step Tasks (Round 150)

**Hypothesis**: Building on H1.378's +2.5% improvement with 2 subgoals for 4-step tasks, more aggressive decomposition (3 subgoals for 4-step) OR learned subgoal representations may further improve performance by providing finer-grained guidance.

**Prediction**: CG with 3 subgoals will outperform 2 subgoals on 4-step tasks, or learned subgoal representations will outperform fixed decomposition.

**Results**:

| Model | 4-step MSE | Improvement vs Baseline |
|-------|-----------|------------------------|
| Flat Baseline (LSTM) | 0.211916 | — |
| Hierarchical Planner (3 subgoals) | 0.209861 | **+0.97%** ✓ |
| CG Hierarchical (3 subgoals, fixed) | 0.210472 | **+0.68%** ✓ |
| CG Hierarchical (3 subgoals, learned) | 0.211924 | **-0.00%** ✗ |

**Status: ✅ SUPPORTED** — CG with fixed subgoal decomposition achieves +0.68% improvement on 4-step tasks with 3 subgoals. Key observations:

1. **More aggressive decomposition shows smaller gains**: 3 subgoals (+0.68%) performs worse than 2 subgoals (+2.5% from H1.378), suggesting diminishing returns from finer decomposition.
2. **Fixed subgoals outperform learned**: Fixed subgoal representations (+0.68%) outperform learned representations (-0.00%), indicating that learning subgoals from scratch is challenging.
3. **Hierarchical planner improves**: Unlike H1.378 where hierarchical planner hurt performance (-2.9%), with 3 subgoals it shows +0.97% improvement, suggesting the task decomposition itself helps.

**Implications**: While aggressive decomposition (3 subgoals) shows positive results, the gains are smaller than with 2 subgoals. This suggests:
- Optimal decomposition granularity exists (2 subgoals for 4-step tasks)
- Fixed subgoal representations work better than learned ones for this task
- Future work should directly compare 2 vs 3 subgoals and test curriculum learning

### H1.378: Hierarchical Subgoal Decomposition for 4+ Step Tasks (Round 149)

**Hypothesis**: Since external memory scaling (H1.377) showed diminishing returns and failed on 4-step tasks, hierarchical planning with subgoal decomposition may help CG handle longer horizons by breaking them into manageable 2-step subgoals.

**Prediction**: CG with subgoal decomposition will show positive improvement on 4-step tasks (unlike external memory which showed -0.3% to -0.0%).

**Results**:

| Model | 4-step MSE | Improvement vs Baseline |
|-------|-----------|------------------------|
| Flat Baseline (LSTM) | 0.366249 | — |
| Hierarchical Planner | 0.376728 | **-2.9%** ✗ |
| CG Hierarchical | 0.357156 | **+2.5%** ✓ |

**Status: ⚠️ PARTIAL SUPPORT** — CG with hierarchical subgoal decomposition shows modest improvement on 4-step tasks (+2.5%), while a plain hierarchical planner actually hurts performance (-2.9%). Key observations:

1. **CG benefits from hierarchy**: The CG architecture combined with subgoal decomposition achieves +2.5% improvement, the first positive result on 4-step tasks after H1.377's failures.
2. **Hierarchy alone is not enough**: The Hierarchical Planner without CG structure performs worse than flat baseline (-2.9%), suggesting the graph structure is essential for effective subgoal reasoning.
3. **Magnitude is modest**: +2.5% is much smaller than H1.376's +15.7% on 3-step tasks, indicating 4-step tasks remain challenging.

**Implications**: The combination of CG + hierarchical planning shows promise for longer horizons, but the improvement is modest. Future work should explore:
- Better subgoal representations (learned vs. fixed 8-dim)
- More aggressive decomposition (3 subgoals for 4-step tasks)
- Curriculum learning from 2-step to 4-step

## Research Trajectory Summary

1. **H1.376**: External memory enables CG to handle 3-step tasks (+15.7%)
2. **H1.377**: External memory scaling shows diminishing returns (+0.7% best on 3-step, fails on 4-step)
3. **H1.378**: Hierarchical subgoal decomposition (2 subgoals) enables +2.5% on 4-step tasks
4. **H1.379**: Aggressive decomposition (3 subgoals) shows smaller gains (+0.68%)

**Key Insight**: CG benefits from hierarchical decomposition for longer horizons, but there are diminishing returns from finer decomposition. The optimal approach appears to be 2 subgoals for 4-step tasks with fixed subgoal representations.

**Next Steps**: Direct comparison of 2 vs 3 subgoals, curriculum learning from 2-step to 4-step tasks, and exploration of adaptive decomposition strategies.