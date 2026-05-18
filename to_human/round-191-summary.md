# Round 191 Summary

## Action Taken

**H1.425: Per-Object CG on Complex Multi-Step Tasks**

Tested whether Per-Object CG's advantage increases with task complexity (number of manipulation stages: 2, 3, 4).

## Results

| Stages | Per-Object vs 2-Node CG |
|--------|------------------------|
| 2 | +60.08% (worse) |
| 3 | +44.68% (worse) |
| 4 | +44.85% (worse) |

**Conclusion**: NOT_SUPPORTED — Per-Object CG performs significantly WORSE than 2-Node CG across all complexity levels. The advantage DECREASES with task complexity, contradicting the hypothesis.

## Key Insight

Per-Object CG's explicit object representation appears to overfit to specific object configurations rather than learning generalizable manipulation patterns. The simpler 2-Node architecture (physical + semantic) is more robust for multi-stage manipulation tasks.

## Next Action

H1.426: Test Per-Object CG on tasks with explicit object relations (spatial relationships between objects).
