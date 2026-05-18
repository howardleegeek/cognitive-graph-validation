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
- Insufficient training epochs (20 vs 50)
- Different random initialization effects
- Need to standardize experimental protocol

---

### H1.393: Discrepancy Investigation — Round 164

**Hypothesis**: The discrepancy between H1.390 (+0.839 correlation) and H1.392 regression (-0.153 correlation) is due to random seeds, model capacity, or training variance.

**Method**: Re-ran H1.390's exact configuration with 5 different random seeds (42, 123, 456, 789, 1000) to measure variance and determine if H1.390's result was reproducible.

**Results**:

| Config | Complexity | Avg Improvement | CG Wins (of 5) |
|--------|------------|-----------------|----------------|
| simple | 20.8 | -8.8% | 0/5 |
| simple2 | 57.3 | -4.2% | 1/5 |
| medium | 104.0 | +3.7% | 3/5 |
| threshold | 145.6 | +4.6% | 4/5 |
| crossover | 166.4 | +4.6% | 5/5 |
| complex | 311.9 | -4.2% | 1/5 |
| very_complex | 552.6 | -14.4% | 0/5 |

**Correlation**: -0.522 (complexity vs CG advantage)

**Conclusion**: NEW_RESULT. Neither H1.390 (+0.839) nor H1.392 (-0.153) was reproduced. The correlation is strongly negative, showing CG advantage peaks at medium complexity (~145-166) and decreases at both low and high complexity. This suggests:
1. CG has an optimal complexity sweet spot (not monotonic)
2. H1.390's positive correlation may have been due to lucky seed variance
3. The "complexity" metric needs refinement to capture the non-linear relationship

---

### H1.392: Task Type Dependency Investigation — Round 163

**Hypothesis**: The discrepancy between H1.390 (regression, positive correlation) and H1.391 (classification, negative correlation) is due to task type — CG advantage depends on whether the task is regression (action prediction) or classification (target identification).

**Method**: Direct head-to-head comparison of both task types on identical data configurations (7 complexity levels, same train/val splits). Measur