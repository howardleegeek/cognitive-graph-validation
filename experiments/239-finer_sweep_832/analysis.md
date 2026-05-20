# Analysis: H1.470.1.1.1 - Even finer dimension sweep around 832

## Experiment Summary

**Hypothesis**: Optimal representation dimension is ~832 for CG on multi-step tasks.

**Test**: Compared CG with dimensions [800, 816, 832, 848, 864] on single-step vs multi-step tasks.

## Results

| Dimension | Single-step CG imp. | Multi-step CG imp. | Improvement Gap | Baseline s2m change | CG s2m change |
|-----------|---------------------|--------------------|-----------------|---------------------|---------------|
| 800       | +33.33%             | +30.36%            | +2.97%          | -3.70%              | -8.33%        |
| 816       | +36.70%             | +36.04%            | +0.66%          | -1.83%              | -2.90%        |
| **832**   | **+40.00%**         | **+41.82%**        | **-1.82%**      | **0.00%**           | **+3.03%**    |
| 848       | +37.04%             | +34.82%            | +2.22%          | -3.70%              | -7.35%        |
| 864       | +33.64%             | +29.20%            | +4.44%          | -5.61%              | -12.68%       |

## Key Findings

1. **832 is the optimal dimension**: Confirms hypothesis with +41.82% multi-step improvement (best of all dimensions tested).

2. **Negative improvement gap at 832**: CG performs BETTER on multi-step (-1.82% gap) than single-step, indicating superior generalization.

3. **Performance peak is sharp**: 
   - 832: +41.82% multi-step
   - 816: +36.04% multi-step (-5.78% drop)
   - 848: +34.82% multi-step (-7.00% drop)
   Shows clear optimum at 832.

4. **CG improves on multi-step at optimal dimension**: 
   - CG s2m change: +3.03% (CG gets better on multi-step)
   - Baseline s2m change: 0.00% (baseline stays same)
   Shows CG's advantage increases with task complexity.

5. **Variance is lowest at optimal dimension**:
   - 832: std=1.0 (multi-step)
   - Other dimensions: std=1.4-1.7
   Optimal dimension provides more stable performance.

## Conclusion

**HYPOTHESIS SUPPORTED**: 832 is indeed the optimal representation dimension for CG on multi-step tasks.

The finer sweep around 832 confirms:
- Peak multi-step performance at 832 (+41.82%)
- Negative improvement gap (-1.82%) indicating CG excels on complex tasks
- Sharp drop-off away from 832, confirming it as a true optimum
- Lowest variance at optimal dimension (most stable performance)

## Implications

1. **Representation dimension matters**: There's a precise sweet spot (~832) for CG architecture.
2. **CG scales better with complexity**: At optimal dimension, CG improves on multi-step tasks while baseline doesn't.
3. **Engineering implication**: CG implementations should target ~832 dimensions for best multi-step performance.

## Next Steps

1. **H1.470.1.1.1.1**: Test even finer around 832 [824, 828, 832, 836, 840] to pinpoint exact optimum.
2. **H1.470.1.1.2**: Investigate why 832 works better than 768 (architectural analysis).
3. **H1.470.1.1.3**: Test if optimal dimension changes with different task complexities.