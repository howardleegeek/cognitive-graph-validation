# Round 150 Summary - Cognitive Graph Autoresearch

**Date**: 2026-05-15  
**Experiment**: H1.379 - Aggressive Subgoal Decomposition for 4+ Step Tasks  
**Status**: ✅ SUPPORTED  

## Key Findings

1. **More aggressive decomposition shows diminishing returns**: Testing 3 subgoals for 4-step tasks (vs 2 subgoals in H1.378) yielded +0.68% improvement with fixed subgoal representations, which is smaller than the +2.5% achieved with 2 subgoals.

2. **Fixed subgoals outperform learned**: Fixed subgoal representations (+0.68%) performed better than learned subgoal representations (-0.00%), suggesting that learning subgoals from scratch is challenging for this task.

3. **Hierarchical planner improves with finer decomposition**: Unlike H1.378 where a hierarchical planner without CG structure hurt performance (-2.9%), with 3 subgoals it showed +0.97% improvement, indicating the task decomposition itself helps.

4. **Optimal granularity appears to be 2 subgoals for 4-step tasks**: The smaller gains from 3 subgoals suggest there's an optimal decomposition granularity, and finer decomposition doesn't necessarily yield better results.

## Implications

The results suggest that while hierarchical decomposition helps Cognitive Graph handle longer-horizon tasks (4+ steps), there are diminishing returns from finer decomposition. The combination of CG structure with appropriate task decomposition (2 subgoals for 4-step tasks) appears optimal. Future work should directly compare 2 vs 3 subgoals and explore curriculum learning approaches.

## Next Steps

The next intended action (H1.380) is to directly compare 2 vs 3 subgoals and test curriculum learning from 2-step to 4-step tasks to see if gradual complexity increase improves performance.