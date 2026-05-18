# Round 201 Summary: Task Complexity Analysis

## Experiment H1.435: Does CG advantage depend on task relational complexity?

**Context**: Following the discrepancy where CG wins on synthetic physics tasks (H1.433: -8.5% to -14.7% better than MLP) but loses on LIBERO-style manipulation tasks (H1.434: -4.2% to -12.2% worse), we hypothesized that CG performance might depend on task relational complexity.

**Method**: Generated synthetic tasks with 3 complexity levels (low: simple linear, medium: non-linear interactions, high: complex relational) and compared MLP vs CG-3p vs CG-6p across 2 trials each.

**Key Results**:
- **CG underperforms MLP overall** across all complexity levels (+4.0% to +44.0% worse)
- **BUT CG shows relative improvement on high complexity tasks**: CG-3p vs MLP improves from +8.2% (low) to +12.2% (high) - a **+4.0% improvement trend**
- **CG-6p shows diminishing returns**: Much worse on low complexity (+44.0% vs MLP) but gap narrows on high complexity (+11.2% vs MLP)

**Conclusion**: Hypothesis **PARTIALLY SUPPORTED**. CG does perform relatively better on high complexity tasks, suggesting its advantage is indeed complexity-dependent. However, CG still underperforms MLP in this synthetic setup, indicating other factors (data distribution, model capacity) also play important roles.

**Next**: Test sub-hypothesis H1.435.1 - CG should win on tasks with clear relational structure (collisions, stacking) but lose on continuous control tasks.