# Round 160 Summary: Complexity Threshold Discovered

**Experiment H1.389** successfully identified a **complexity threshold** where Cognitive Graph starts outperforming the baseline. Testing across 1-10 objects, we found:

- **Crossover at 8 objects**: Below this, baseline wins; above, CG wins
- **Strong correlation (0.837)** between task complexity and CG advantage
- **CG advantage scales**: From -46% at 1 object to +10.16% at 10 objects
- **Representation size matters**: Small CG wins at threshold, large CG wins at higher complexity

This resolves the H1.386/H1.387 discrepancy: different datasets had different complexity levels. The unified representation has overhead that only pays off above a certain complexity threshold. Next step: predict threshold from dataset statistics (H1.390).