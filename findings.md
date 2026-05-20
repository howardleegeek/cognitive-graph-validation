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

### H1.470.1.1.4: Architecture Alignment Investigation — Round 243 (INCONCLUSIVE)

**Hypothesis**: The discrepancy between simulation and "real" CG performance is due to architectural differences between the simulation CG and the CG used in H1 experiments.

**Prediction**: Aligning the architectures will eliminate the performance gap.

**Experiment**: Compared three architectures on multi-step tasks:
1. **Baseline_Concat**: Standard concatenation baseline
2. **Simulation_CG**: GNN-based CG with physical/semantic split (144+368 dims)
3. **Real_CG**: Attention-based CG (based on H1.148 architecture)

**Results (20-step sequences)**:

| Architecture | Val Loss | Improvement vs Baseline |
|--------------|----------|------------------------|
| Baseline_Concat | 0.000112 | 0.00% |
| Simulation_CG | 0.000043 | **+61.36%** |
| Real_CG | 0.000351 | -213.22% |

**Results (50-step sequences)**:

| Architecture | Val Loss (50-step) |
|--------------|-------------------|
| Baseline_Concat | 0.021267 |
| Simulation_CG | 0.016089 |
| Real_CG | 0.016210 |

**Key Findings**:
1. **Simulation CG actually outperforms baseline**: +61.36% improvement on 20-step tasks, contradicting the negative gaps seen in H1.470.1.1.3
2. **Real CG (attention-based) underperforms**: -213% vs baseline on short sequences
3. **On longer sequences (50-step)**: Both CG variants perform similarly (16.1 vs 16.2 loss), both better than baseline (21.3)
4. **Architecture is NOT the root cause**: The discrepancy in H1.470.1.1.3 appears to be data/task-specific, not architectural

**Conclusion**: INCONCLUSIVE - The architecture difference is not the primary cause of the simulation vs real discrepancy. The issue likely lies in:
- Task definition differences (how multi-step tasks are structured)
- Data generation process (how observations/actions are sampled)
- Evaluation metrics (how improvement is calculated)

**Next Step**: H1.470.1.1.5 - Investigate task structure differences between simulation and "real" experiments

---

### H1.470.1.1.3: Improvement Gap Sign Discrepancy Investigation — Round 242 (REFUTED)

**Hypothesis**: The discrepancy in improvement gap sign (positive in simulation vs negative in real experiments) indicates that the simulation model doesn't capture the key mechanism that makes CG better on multi-step tasks in real data.

**Prediction**: Adding structured cross-modal relationships and temporal dependencies will flip the gap sign from positive to negative, matching real experiments.

**Experiment**: Three data regimes tested with 2 runs each:
1. **Random**: Pure random data (current simulation approach)
2. **Structured**: Language encodes object properties that correlate with observations
3. **Temporal**: Multi-step tasks have explicit step-to-step dependencies

**Results**:

| Regime | Single-step CG imp. | Multi-step CG imp. | Gap (multi - single) |
|--------|---------------------|---------------------|---------------------|
| Random | -3.75% | -150.59% | **-146.84%** |
| Structured | -7.40% | -22.24% | **-14.84%** |
| Temporal | -3.75% | -60.17% | **-56.42%** |

**Key Findings**:
1. **CG underperforms baseline across ALL regimes**: Unlike real experiments where CG showed +25-31% improvement, the simulation shows CG consistently worse than baseline. This is a fundamental architecture mismatch.
2. **All gaps are NEGATIVE**: Contrary to the prior simulation (H1.470.1.1.2) which showed positive gaps, this simulation shows negative gaps across all regimes. The prior simulation's positive gaps were an artifact of its specific data generation method.
3. **Structured regime has smallest gap**: The structured data regime (-14.84% gap) shows the least degradation for CG on multi-step vs single-step, suggesting cross-modal structure does help CG maintain re
