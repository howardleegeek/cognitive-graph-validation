# Round 241 Summary: H1.470.1.1.2 - Task Complexity Stability

## Experiment Executed
Tested whether the optimal representation dimension (816 from H1.470.1.1.1) is stable across different task complexities (2-step, 4-step, 5-step) or shifts with sequence length.

## Key Finding
**Hypothesis REFUTED**: Optimal dimension does NOT increase monotonically with task complexity. The pattern is non-monotonic:
- 2-step tasks: optimal at 800 dimensions (25.08% improvement)
- 3-step tasks: optimal at 832 dimensions (30.88% improvement)  
- 4-step tasks: optimal at 816 dimensions (26.69% improvement)
- 5-step tasks: optimal at 896 dimensions (26.54% improvement)

## Important Discrepancy
The simulation showed **positive improvement gaps** (CG better on single-step tasks), contradicting H1.470.1.1.1 which found **negative gaps** (CG better on multi-step tasks). This suggests the simulation may not capture CG's true multi-step advantage.

## Implications
1. **816 is not universally optimal**: While best for 4-step tasks in this simulation, different complexities prefer different dimensions.
2. **No clear complexity-dimension relationship**: The pattern is non-monotonic, suggesting a more complex relationship.
3. **Need real experiment validation**: The gap sign discrepancy indicates simulation limitations.

## Next Step
H1.470.1.1.3: Investigate why improvement gaps are positive in simulation vs negative in real experiments to understand if simulation captures CG's multi-step advantage.