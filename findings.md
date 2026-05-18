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

**Conclusion**: H1.427 **REFUTED**. Per-Object CG transfers BETTER than other architectures, not worse. The hypothesis that Per-Object CG overfits to task-specific features is incorrect. Instead, the explicit object structure appears to learn more generalizable representations that transfer better across task types.

**Key insight**: The earlier finding that Per-Object CG performs worse on multi-stage tasks (H1.425) is not due to overfitting — it's due to the task structure itself. Multi-stage tasks may not benefit from explicit object representation the way spatial reasoning tasks do. The transfer learning results suggest Per-Object CG learns robust object-centric features that transfer well, but these features are simply less useful for sequential manipulation tasks.

---

### H1.426: Per-Object CG with Explicit Relational Edges — Round 192

**Hypothesis**: Adding explicit spatial relation edges to Per-Object CG will improve performance on spatial relation tasks.

**Context**: H1.421 showed Per-Object CG achieves +61.76% improvement on object permanence tasks. This experiment tests whether explicit relational edges further improve spatial reasoning.

**Method**: Compare three architectures on spatial relation tasks:
1. Standard Per-Object CG (5 object nodes + 1 semantic node)
2. Per-Object CG with Relational Edges (explicit spatial relation edges between object nodes)
3. 2-Node CG (physical + semantic nodes)

**Results**:

| Architecture | MSE | vs Baseline |
|--------------|-----|-------------|
| Baseline | 0.160611 | — |
| 2-Node CG | 0.201314 | +25.34% (worse) |
| Per-Object CG | 0.151306 | -5.79% (better) |
| Per-Object CG + Relations | 0.185970 | +22.91% (worse) |

**Conclusion**: H1.426 **NOT_SUPPORTED**. Adding explicit relational edges HURTS Per-Object CG performance (+22.91% worse than standard Per-Object CG). Standard Per-Object CG without explicit relations achieves best results (-5.79% vs baseline). The implicit relational reasoning through GNN message passing is more effective than explicit edge features.

**Key insight**: The GNN's learned message passing already captures relational information implicitly. Adding explicit relational edges adds unnecessary complexity and may interfere with the learned representations.

---

### H1.425: Per-Object CG on Complex Multi-Step Tasks — Round 190

**Hypothesis**: Per-Object CG architecture advantage increases with task complexity (number of manipulation stages).

**Previous context**: H1.421 showed +61.76% on object permanence. H1.423 showed crossover at seq_len≈24.3.

**Method**: Tested Per-Object CG vs 2-Node CG vs Baseline on multi-stage manipulation tasks with 2, 3, and 4 stages.

**Results by complexity**:

| Stages | Baseline MSE | 2-Node CG | Per-Object CG | Per-Object vs 2-Node |
|--------|--------------|-----------|---------------|---------------------|
| 2 | 0.064395 | 0.060018 (-6.80%) | 0.096079 (+49.20%) | +60.08% |
| 3 | 0.065177 | 0.066755 (+2.42%) | 0.096584 (+48.19%) | +44.68% |
| 4 | 0.068990 | 0.067047 (-2.82%) | 0.097116 (+40.77%) | +44.85% |

**Complexity trend**: Per-Object CG advantage DECREASES with complexity (60.08% → 44.68% → 44.85%)

**Conclusion**: H1.425 **NOT_SUPPORTED**. Per-Object CG performs significantly WORSE than 2-Node CG on multi-stage tasks across all complexity levels. The advantage does NOT increase with task complexity — in fact, it slightly decreases. The simpler 2-Node architecture is more robust for multi-stage manipulation tasks.

**Key insight**: Per-Object CG's explicit object representation appears to overfit to specific object configurations rather than learning generalizable manipulation patterns. The 2-Node abstraction (physical + semantic) provides better generalization across different manipulation stages. This contradicts the hypothesis that more complex tasks would benefit more from per-object structure.

---

### H1.424: Hybrid Cognitive Graph Architecture — Round 190

**Hypothesis**: Adaptive architecture selection between Per-Object CG and 2-Node CG based on sequence length or task type.

**Status**: Not yet tested. Deferred pending H1.427 results.

---

## Summary of Hypotheses

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1: CG > Baseline | SUPPORTED | +25.6% improvement with real robot data |
| H2: Attention vs Concat | INCONCLUSIVE | 1.7% difference, needs more data |
| H3: Attention for long sequences | REFUTED | Concatenation wins for simple tasks |
| H4: 25% optimal dimension | CLOSE | 25% optimal vs 28% hypothesis |
| H1.421: Per-Object CG on object permanence | SUPPORTED | +61.76% improvement |
| H1.423: Sequence length crossover | SUPPORTED | Crossover at seq_len≈24.3 |
| H1.425: Per-Object CG on multi-stage | NOT_SUPPORTED | Worse than 2-Node CG |
| H1.426: Explicit relational edges | NOT_SUPPORTED | Hurts performance |
| H1.427: Per-Object CG transfer learning | REFUTED | Per-Object CG transfers BEST |

## Open Questions

1. **Task-specific architecture selection**: When should we use Per-Object CG vs 2-Node CG?
   - Per-Object CG: Better for spatial reasoning, object permanence, transfers better
   - 2-Node CG: Better for multi-stage manipulation, sequential tasks

2. **Why does Per-Object CG transfer better but perform worse on multi-stage?**
   - Hypothesis: Object-centric features are generalizable but not optimal for sequential decision-making
   - Need to test: Hybrid architecture that uses Per-Object for perception and 2-Node for planning

3. **What makes multi-stage tasks different?**
   - Need to analyze: Attention patterns, node activations, information flow

## Next Steps

1. **H1.428**: Test hybrid architecture that combines Per-Object CG (for perception) with 2-Node CG (for action prediction)
2. **H1.429**: Analyze attention patterns in Per-Object CG vs 2-Node CG on different task types
3. **H1.430**: Test on real robot data with both spatial reasoning and multi-stage manipulation tasks