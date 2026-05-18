# Research Findings — Cognitive Graph Architecture

## Research Question

Does a unified cognitive graph architecture (early fusion of physical and semantic representations) achieve higher sample efficiency than separated architectures (JEPA + LLM alignment) on language-conditioned robotic tasks?

## Current Understanding

**The Core Hypothesis**: Current approaches (V-JEPA 2, π0, LED-WM) all suffer from representation separation — vision and language exist in different spaces and are only aligned after encoding. This causes:
1. **Grounding problems**: Language not truly grounded in physical dynamics
2. **Combinatorial explosion**: Need to learn all (vision, language) pairings separately
3. **Learning inefficiency**: No gradient flow between modalities during training

**The Cognitive Graph Solution**: A unified 512-dimensional representation space where:
- 144 dimensions encode physical world state (analogous to V-JEPA embeddings)
- 368 dimensions encode semantic/conceptual information (analogous to LLM embeddings)
- Single GNN processes both, with cross-modal attention allowing dynamic interaction
- Explicit graph structure (nodes = objects/concepts, edges = relationships/physics)

## Key Results

### H1.395: Protocol Standardization — Round 166

**Hypothesis**: The discrepancy between H1.393 (CG wins at medium complexity) and H1.394 (CG loses everywhere) is due to differences in data generation or training parameters, not a fundamental finding.

**Method**: Ran both H1.393 and H1.394 style experiments with identical seeds (42), data generation, and training parameters (20 epochs). Also ran a unified experiment covering all complexity levels.

**Results**:

| Style | Correlation | Avg Improvement | CG Wins |
|-------|-------------|-----------------|---------|
| H1.393 (7 configs) | -0.621 | -3.2% | 1/7 |
| H1.394 (8 configs) | -0.506 | -5.0% | 1/8 |
| UNIFIED (10 configs) | -0.552 | -4.5% | 1/10 |

**Detailed Results (UNIFIED style)**:
| Complexity | Improvement | CG Wins |
|------------|-------------|---------|
| 20 | -2.0% | No |
| 60 | -2.8% | No |
| 100 | +0.7% | Yes |
| 150 | -2.5% | No |
| 170 | -2.6% | No |
| 200 | -7.2% | No |
| 300 | -8.0% | No |
| 400 | -10.3% | No |
| 500 | -6.1% | No |
| 600 | -4.1% | No |

**Conclusion**: DISCREPANCY_RESOLVED. Both H1.393 and H1.394 styles now show similar negative correlations (-0.5 to -0.6), confirming:
1. The original H1.393 result (positive correlation) was likely a seed artifact
2. CG underperforms baseline across most complexity levels in synthetic data
3. Only at complexity=100 does CG show slight advantage (+0.7%)
4. The inverted-U pattern is NOT confirmed - instead there's a negative linear relationship

**Key Insight**: The CG architecture as currently implemented struggles with this synthetic data task. The unified representation may be overkill for simple pattern learning, or the architecture needs tuning for this specific task type.

### H1.394: Quadratic Complexity Relationship — Round 165

**Hypothesis**: CG advantage follows an inverted-U (quadratic) relationship with task complexity, peaking at medium complexity (~145-166) and decreasing at both low and high complexity.

**Method**: Tested 8 complexity levels (50-600) with 2 seeds each. Fit both linear and quadratic models to CG advantage vs complexity. Compared model fits using AIC.

**Results**:

| Target Complexity | Actual Complexity | Avg Improvement | CG Wins |
|-------------------|-------------------|-----------------|---------|
| 50 | 120.5 | -25.97% | 0/2 |
| 100 | 131.0 | -24.45% | 0/2 |
| 150 | 202.5 | -21.11% | 0/2 |
| 200 | 198.0 | -25.97% | 0/2 |
| 300 | 287.0 | -26.32% | 0/2 |
| 400 | 364.0 | -26.32% | 0/2 |
| 500 | 427.5 | -37.65% | 0/2 |
| 600 | 455.5 | -38.12% | 0/2 |

**Model Comparison**:
- Linear Model: R² = 0.6606, AIC = 23.48
- Quadratic Model: R² = 0.8918, AIC = 16.33
- ΔAIC (quad - linear) = -7.15 (quadratic better)
- Peak complexity = 214.8

**Conclusion**: PARTIALLY_SUPPORTED. Quadratic model fits significantly better than linear (ΔAIC = -7.15, ΔR² = +0.23), confirming the inverted-U pattern. However, the peak at complexity ~215 is higher than the predicted 150-170 range from H1.393. This suggests:
1. The inverted-U relationship is real, not a seed artifact
2. The optimal complexity range may be broader than initially estimated
3. CG architecture struggles across all tested complexity levels in this experiment (all negative improvements)
4. Need to investigate why CG underperforms baseline in synthetic data

**Key Insight**: The discrepancy between H1.393 (CG wins at medium complexity) and H1.394 (CG loses everywhere) may be due to:
- Different data generation methods
- Insufficient training epochs (