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

### H1.428: Hybrid Architecture — Round 194

**Hypothesis**: A hybrid architecture combining Per-Object CG (for perception) with 2-Node CG (for action prediction) will outperform both individual architectures.

**Context**: H1.427 showed Per-Object CG transfers best across task types. This experiment tests whether combining the strengths of both architectures yields better performance.

**Method**: Train Baseline, Per-Object CG, 2-Node CG, and Hybrid (fuses both) on two task types. Compare validation MSE.

**Results**:

| Architecture | spatial_relations MSE | multi_stage MSE | Avg MSE |
|--------------|----------------------|-----------------|---------|
| Baseline | 1.111649 | 1.059912 | 1.085781 |
| Per-Object CG | 1.138354 | 1.088253 | 1.113304 |
| 2-Node CG | 1.139350 | 1.125620 | 1.132485 |
| Hybrid | 1.133744 | 1.145911 | 1.139827 |

**Key Findings**:
1. **Baseline outperforms all CG variants** on this synthetic task (no structured signal to exploit)
2. Hybrid (1.139827) does NOT beat best individual (Per-Object: 1.113304)
3. Per-Object CG performs best among CG variants
4. The additional complexity of hybrid architecture adds overhead without benefit

**Conclusion**: H1.428 **NOT_SUPPORTED**. The hybrid architecture does not outperform individual architectures. Per-Object CG alone remains the best CG variant.

---

### H1.427: Task Type Transfer Learning — Round 193

**Hypothesis**: Per-Object CG learns task-specific features that don't transfer well, while 2-Node CG learns more generalizable representations.

**Context**: H1.425 showed Per-Object CG performs worse on multi-stage tasks. H1.426 showed explicit relational edges hurt performance. This experiment investigates whether Per-Object CG overfits to task-specific features.

**Method**: Train each architecture on one task type, evaluate zero-shot transfer to another task type. Measure transfer gap = (transfer_mse - source_mse) / source_mse.

**Task Types**:
- Spatial relations: Object permanence / spatial reasoning (5 objects, relation queries)
- Multi-stage: Sequential manipulation with 3 stages

**Results**:

| Architecture | Spatial→Multi Transfer Gap | Multi→Spatial Transfer Gap | Average Transfer Gap |
|--------------|---------------------------|---------------------------|---------------------|
| Baseline | +105.38% | +8504.77% | +4305.08% |
| 2-Node CG | +90.34% | +3424.14% | +1757.24% |
| Per-Object CG | +75.66% | +2078.17% | +1076.91% |

**Key Findings**:
1. **Per-Object CG transfers BEST** — lowest average transfer gap (+1076.91% vs +1757.24% for 2-Node CG vs +4305.08% for Baseline)
2. All architectures struggle with multi_stage→spatial transfer (massive gaps 2000-8500%)
3. Spatial→Multi transfer is much more manageable (75-105% gaps)
4. Fine-tuning recovers performance quickly (negative gaps for spatial→multi direction)

**Conclusion**: H1.427 **REFUTED**. Per-Object CG transfers BETTER than other architectures, not worse. The hypothesis that Per-Object CG overfits to task-specific features is incorrect. Instead, the explicit object structure appears to learn more generalizable representations that transfer across task types.
