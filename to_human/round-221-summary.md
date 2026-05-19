# Round 221 Summary

**Experiment**: H1.455 - Sub-goal Generalization Across Task Complexities

**Hypothesis**: The optimal 3 sub-goals from H1.454 will generalize across different task complexities (varying steps per sub-goal: 2/3/5).

**Result**: **REFUTED** - The optimal 3 sub-goals from H1.454 do NOT generalize across task complexities. CG loses to baseline at all tested complexity levels:
- 2 steps/subgoal: -0.18% (Baseline wins)
- 3 steps/subgoal: -0.10% (Baseline wins)  
- 5 steps/subgoal: -2.16% (Baseline wins)
- Average: -0.81%

**Key Insight**: Task-dependent optimality confirmed. The relationship between sub-goal granularity and performance is task-dependent - what works for one task complexity does not work for another. This explains why H1.453's massive +82.81% gain hasn't replicated in subsequent experiments.

**Next**: H1.456 will investigate why H1.453 showed massive gains while subsequent experiments show marginal/negative results.
