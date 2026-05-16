# Sub-Hypothesis H1.369: Autocorrelation Threshold for CG Effectiveness

## Parent Hypothesis
H1: Early fusion wins on real robot data (+25.6%)

## Sub-Hypothesis Statement
There exists a critical autocorrelation threshold ρ* ≈ 0.5-0.6 above which Cognitive Graph architecture significantly outperforms baseline (≥15% improvement), and below which it underperforms or matches baseline.

## Concrete Prediction
| Autocorrelation (ρ) | Predicted CG Improvement |
|---------------------|-------------------------|
| 0.0 - 0.3          | -5% to +5% (noise)      |
| 0.4 - 0.5          | +5% to +15% (transition)|
| 0.6 - 0.7          | +15% to +30% (moderate) |
| 0.8 - 0.9          | +30% to +50% (strong)   |
| 0.95+              | +50% to +90% (very strong)|

## Falsifiability Criterion
- **Supported**: Clear monotonic trend with ρ* identified between 0.4-0.6
- **Refuted**: No clear threshold, or threshold outside predicted range, or non-monotonic relationship

## Test Plan
1. Generate synthetic datasets with controlled autocorrelation: ρ ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95}
2. Train both Baseline and Cognitive Graph on each autocorrelation level
3. Measure improvement percentage for each ρ
4. Fit sigmoid to identify ρ* (inflection point)
5. Verify monotonic relationship

## Expected Outcome
Based on H1.181 (attention) and H1.367-H1.368 (CG with autocorrelation), we expect:
- ρ* ≈ 0.55 ± 0.1
- Clear sigmoid relationship between ρ and CG improvement
- This explains why real robot data (ρ≈0.7-0.95) shows strong CG advantage

## Why This Matters
1. **Practical**: Determines which datasets/tasks benefit from CG architecture
2. **Theoretical**: Links temporal structure to representation learning
3. **Actionable**: Provides decision criterion for architecture selection

## Related Experiments
- H1.181: Attention advantage increases with autocorrelation (-6.5% to -26.9%)
- H1.367: CG +85.7% on 20-40 steps with autocorrelation
- H1.368: CG +90.5% on 30-50 steps with high autocorrelation