# Round 265 Summary — Hierarchical Multi-Step Tasks with Phase Transitions

## Experiment: H1.470.1.1.26

**Task**: Test ensemble disagreement noise estimation on hierarchical multi-step tasks with phase transitions (approach → grasp → transport).

**Result**: **REFUTED** ❌

| Strategy | Test Loss | Improvement | Oracle Ratio |
|----------|-----------|-------------|--------------|
| Baseline | 0.4438 | +0.00% | N/A |
| Oracle noise | 0.4233 | +4.62% | 100% |
| Ensemble disagreement | 0.4618 | **-4.05%** | -87.7% |

**Key Finding**: Ensemble disagreement performs **worse** than baseline on hierarchical multi-step tasks with phase transitions. This contrasts sharply with H1.470.1.1.24 (SUPPORTED, +15.24% on real robot data) and H1.470.1.1.25 (INCONCLUSIVE, +0.76% on simple multi-step).

**Interpretation**: The success of ensemble disagreement appears to depend on having sufficient complexity/diversity in the data. Simple hierarchical structures (3 phases) don't provide enough signal for ensemble disagreement to outperform. Oracle noise (+4.62%) outperforms ensemble disagreement, suggesting true noise levels are more informative than model uncertainty on these tasks.

**Next Action**: Test on tasks with more phases (4-5) or explore different uncertainty estimation methods.
