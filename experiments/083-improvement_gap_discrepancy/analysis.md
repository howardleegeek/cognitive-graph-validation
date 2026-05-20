# H1.470.1.1.3: Improvement Gap Sign Discrepancy — Analysis

## Round 242

### Hypothesis
The discrepancy in improvement gap sign (positive in simulation vs negative in real experiments) indicates that the simulation model doesn't capture the key mechanism that makes CG better on multi-step tasks in real data.

### Prediction
Adding structured cross-modal relationships and temporal dependencies will flip the gap sign from positive to negative, matching real experiments.

### Experiment Design
Three data regimes tested with 2 runs each:
1. **Random**: Pure random data (current simulation approach)
2. **Structured**: Language encodes object properties that correlate with observations
3. **Temporal**: Multi-step tasks have explicit step-to-step dependencies

### Results

| Regime | Single-step improvement | Multi-step improvement | Gap (multi - single) |
|--------|------------------------|------------------------|---------------------|
| Random | -3.75% | -150.59% | **-146.84%** |
| Structured | -7.40% | -22.24% | **-14.84%** |
| Temporal | -3.75% | -60.17% | **-56.42%** |

### Key Findings

1. **CG underperforms baseline across ALL regimes**: Unlike real experiments where CG showed positive improvement (+25-31%), the simulation shows CG consistently worse than baseline. This is a fundamental architecture mismatch.

2. **All gaps are NEGATIVE**: Contrary to the prior simulation (H1.470.1.1.2) which showed positive gaps, this simulation shows negative gaps across all regimes. This suggests the prior simulation's positive gaps were an artifact of its specific data generation method.

3. **Structured regime has smallest gap**: The structured data regime (-14.84% gap) shows the least degradation for CG on multi-step vs single-step, suggesting that cross-modal structure does help CG maintain its relative performance on multi-step tasks.

4. **No gap sign flip**: The hypothesis that adding structure would flip the gap sign is REFUTED. All regimes show negative gaps.

### Interpretation

The fundamental issue is that the simulation's CG architecture (with its specific GNN + attention design) is not capturing the mechanism that makes CG superior in real experiments. Possible explanations:

1. **Architecture mismatch**: The simulation CG may be too shallow (2 GNN layers) or have insufficient capacity compared to the real CG architecture
2. **Training dynamics**: The simulation uses simple MSE loss with Adam, while real experiments may use more sophisticated training
3. **Data complexity**: Real robot data has rich physical structure that the simulation's synthetic data cannot replicate

### Conclusion: REFUTED

The hypothesis that simulation data regime explains the gap sign discrepancy is REFUTED. The simulation consistently shows CG underperforming baseline, which is the opposite of real experiments. This suggests the discrepancy is not in the data but in the architecture or training methodology.

### Next Steps
- H1.470.1.1.4: Investigate whether the simulation CG architecture matches the real CG architecture used in H1 experiments
- Alternative: Accept that simulation cannot replicate real CG advantage and focus on real-data experiments only
