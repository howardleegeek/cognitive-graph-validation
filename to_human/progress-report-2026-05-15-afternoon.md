# Cognitive Graph Research Progress Report
## May 15, 2026 - Afternoon Session

### Executive Summary

**Total Experiments Run**: 36 (experiments 365-400)
**Success Rate**: 91.7% (33/36 experiments SUPPORTED)
**Average Improvement**: +20.3%

### Hypothesis Status

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1: Unified vs Baseline | ✅ SUPPORTED | +25.6% with real robot data |
| H2: Explicit Graph | ⚠️ INCONCLUSIVE | 1.7% difference (noise) |
| H3: Attention vs Concat | ❌ REFUTED | Concatenation wins on simple tasks |
| H4: Dimension Allocation | 🔸 CLOSE | 25% optimal vs 28% hypothesis |

### New Experimental Results (365-400)

#### Key Findings:

1. **Cognitive Graph consistently outperforms baseline** - 91.7% win rate
2. **Multi-step tasks show strong results** - +18-31% improvement
3. **Larger scale experiments show high variance** - Best: +40.4%, Worst: -16.3%
4. **Attention complexity tasks perform well** - +13-34% improvement
5. **Longer sequences (20 steps) work** - +6-21% improvement

#### Refutations Analysis:

3 experiments (368, 397, 400) showed the baseline outperforming Cognitive Graph:
- All 3 refutations occurred at larger scales (800+ training samples)
- This suggests the CG advantage is most pronounced at medium scales (200-400 samples)
- At very large scales, the baseline may catch up or outperform

### Research Trajectory

The autonomous research engine continues to run experiments automatically. Each experiment:
1. Generates a hypothesis based on previous results
2. Runs the experiment
3. Updates findings.md
4. Commits to GitHub
5. Generates progress report

### Next Steps

Based on the current findings:
1. **Deepen H1 success**: Test with more complex multi-step tasks
2. **Address H3 failure**: Test attention on longer sequences (20+ timesteps)
3. **Investigate scaling**: Understand why large-scale experiments sometimes fail
4. **Generate new sub-hypotheses**: H1.1, H1.2, H3.1 variants

### Git Status

All experiments have been automatically committed and pushed to GitHub.