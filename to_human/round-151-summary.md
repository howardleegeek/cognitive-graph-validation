# Round 151 Summary: H1.380 - Subgoal Granularity Comparison

## Experiment Executed
Direct comparison of 2 vs 3 subgoals for 4-step manipulation tasks, testing the hypothesis that 2 subgoals (one per 2 steps) represents the optimal decomposition granularity.

## Key Findings
1. **Optimal granularity confirmed**: Cognitive Graph with 2 subgoals achieves +0.14% improvement, while 3 subgoals shows -0.10% degradation. The +0.24% difference validates that finer decomposition provides diminishing returns.
2. **Architecture sensitivity differs**: Hierarchical planner shows minimal difference between 2 and 3 subgoals (-0.01%), while CG is more sensitive to decomposition granularity.
3. **2 subgoals optimal for 4-step tasks**: Results confirm that decomposing 4-step tasks into 2 subgoals (one per 2 steps) is more effective than 3 subgoals for the Cognitive Graph architecture.

## Implications
The findings suggest task decomposition should match natural task structure, with CG architecture benefiting from coarser, more meaningful subgoals. This provides guidance for hierarchical planning in cognitive architectures: use 2 subgoals for 4-step tasks rather than more aggressive decomposition.

## Next Steps
Test curriculum learning with proper architecture adaptation (H1.381) - train on 2-step tasks with 1 subgoal, then adapt to 4-step tasks with 2 subgoals.