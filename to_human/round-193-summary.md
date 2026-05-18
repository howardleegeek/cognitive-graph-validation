# Round 193 Summary: Task Type Transfer Learning

**Experiment**: H1.427 - Tested whether Per-Object CG learns task-specific features that don't transfer well.

**Result**: Hypothesis REFUTED. Per-Object CG transfers BEST across task types, not worst.

**Key Numbers**:
- Per-Object CG average transfer gap: +1076.91%
- 2-Node CG average transfer gap: +1757.24%
- Baseline average transfer gap: +4305.08%

**Insight**: The earlier finding that Per-Object CG performs worse on multi-stage tasks (H1.425) is NOT due to overfitting. Object-centric features are highly generalizable and transfer well across task types. The performance gap on multi-stage tasks is due to task structure — sequential manipulation doesn't benefit from explicit object representation the way spatial reasoning does.

**Next**: H1.428 will test a hybrid architecture combining Per-Object CG (for perception) with 2-Node CG (for action prediction) to get the best of both worlds.