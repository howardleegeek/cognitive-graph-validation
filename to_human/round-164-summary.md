# Round 164 Summary

**H1.393 - Discrepancy Investigation**

**What we did**: Investigated why H1.390 showed +0.839 correlation between complexity and CG advantage, while H1.392 regression showed -0.153. Re-ran H1.390's exact configuration with 5 different random seeds (42, 123, 456, 789, 1000) to measure variance and test reproducibility.

**Result**: NEW_RESULT — Neither H1.390 (+0.839) nor H1.392 (-0.153) was reproduced. The new correlation is **-0.522**, showing an inverted-U pattern: CG advantage peaks at medium complexity (~145-166, where CG wins 4-5/5 seeds) and decreases at both low complexity (0/5 wins) and high complexity (0/5 wins). This reveals that the relationship between complexity and CG advantage is non-monotonic, not the simple positive correlation H1.390 suggested.

**Key insight**: The original H1.390 result may have been due to favorable random seed variance. The true pattern is that CG has an optimal complexity sweet spot — too simple and there's no multi-object reasoning to leverage, too complex and the unified representation overhead hurts performance.

**Next**: H1.394 - Investigate the non-linear complexity relationship and develop a refined complexity metric that captures the inverted-U pattern.
