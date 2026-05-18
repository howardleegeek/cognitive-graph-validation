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

**Method**: Direct head-to-head comparison of both task types on identical data configurations (7 complexity levels, same train/val splits). Measured correlation between complexity and CG advantage for each task type.

**Results — Regression (Action Prediction)**:

| Config | Objects | Seq | Complexity | Baseline Loss | CG Loss | Improvement | Winner |
|--------|---------|-----|------------|---------------|---------|-------------|--------|
| simple | 3 | 10 | 21.3 | 0.0166 | 0.0147 | +11.4% | **cg** |
| simple2 | 4 | 15 | 32.7 | 0.0155 | 0.0160 | -3.2% | baseline |
| medium | 5 | 20 | 46.0 | 0.0181 | 0.0169 | +6.5% | **cg** |
| threshold | 6 | 25 | 61.1 | 0.0140 | 0.0149 | -6.0% | baseline |
| crossover | 7 | 30 | 78.0 | 0.0155 | 0.0177 | -14.2% | baseline |
| complex | 8 | 35 | 96.6 | 0.0177 | 0.0175 | +1.2% | **cg** |
| very_complex | 10 | 40 | 131.5 | 0.0156 | 0.0148 | +5.1% | **cg** |

**Results — Classification (Target Object Prediction)**:

| Config | Objects | Seq | Complexity | Baseline Acc | CG Acc | Improvement | Winner |
|--------|---------|-----|------------|--------------|--------|-------------|--------|
| simple | 3 | 10 | 21.3 | 0.44 | 0.44 | +0.0% | baseline |
| simple2 | 4 | 15 | 32.7 | 0.22 | 0.34 | +54.5% | **cg** |
| medium | 5 | 20 | 46.0 | 0.24 | 0.20 | -16.7% | baseline |
| threshold | 6 | 25 | 61.1 | 0.14 | 0.10 | -28.6% | baseline |
| crossover | 7 | 30 | 78.0 | 0.12 | 0.22 | +83.3% | **cg** |
| complex | 8 | 35 | 96.6 | 0.10 | 0.14 | +40.0% | **cg** |
| very_complex | 10 | 40 | 131.5 | 0.08 | 0.10 | +25.0% | **cg** |

**Correlation (Regression)**: -0.153 (weak negative)
**Correlation (Classification)**: +0.560 (moderate positive)

**Conclusion**: INCONCLUSIVE. Task type alone does NOT explain the discrepancy between H1.390 and H1.391. Neither matches H1.390's +0.839. Classification shows interesting pattern: CG wins at higher complexity (last 3 configs all CG wins).
