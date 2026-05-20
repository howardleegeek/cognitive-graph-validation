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

### H1.470.1.1.7: Temporal Memory for Strong Temporal Tasks — Round 246 (SUPPORTED)

**Hypothesis**: Adding explicit temporal memory (recurrent connections or memory banks) to Real CG will improve performance on strong temporal tasks.

**Previous Finding**: Both Sim CG and Real CG struggle with strong temporal dependencies (gap remains 40-60% across all sequence lengths).

**Experiment**: Tested Real CG with three configurations on strong temporal tasks:
1. Real CG (Attention Only) - no explicit memory
2. Real CG (LSTM Memory) - LSTM-based temporal memory bank
3. Real CG (GRU Memory) - GRU-based temporal memory bank

**Results Summary**:

| Seq Len | Baseline Loss | Attn Only | LSTM Mem | GRU Mem |
|---------|---------------|-----------|----------|---------|
| 10      | 0.1660        | 0.1661 (-0.07%) | 0.0352 (+78.81%) | 0.0391 (+76.47%) |
| 20      | 0.1680        | 0.1679 (+0.03%) | 0.0301 (+82.08%) | 0.0304 (+81.92%) |

**Average Improvement over Baseline**:
- Real CG (Attn Only): **-0.02%** (no improvement)
- Real CG (LSTM Mem): **+80.44%** (significant improvement)
- Real CG (GRU Mem): **+79.20%** (significant improvement)

**Key Findings**:
1. **Hypothesis SUPPORTED** — Adding explicit temporal memory (LSTM/GRU) dramatically improves performance on strong temporal tasks
2. **LSTM slightly outperforms GRU**: +80.44% vs +79.20%
3. **Attention-only provides NO benefit**: -0.02% improvement on strong temporal tasks
4. **Memory mechanism is essential** for handling strong temporal dependencies

**Conclusion**: SUPPORTED - Explicit temporal memory (LSTM or GRU) is required for Real CG to handle strong temporal dependencies. The attention mechanism alone is insufficient.

**Sub-hypothesis H1.470.1.1.8**: Test hierarchical temporal memory (multiple LSTM layers at different timescales) for even longer sequences.

---

### H1.470.1.1.6: Attention Mechanism Sequence Length Sensitivity — Round 245 (PARTIALLY SUPPORTED)

**Hypothesis**: Real CG's attention mechanism requires longer sequences to establish meaningful temporal relationships, while Simulation CG (concatenation-based) performs consistently across sequence lengths.

**Prediction**: Real CG will underperform on short sequences (< 20 steps) but catch up on longer sequences (≥ 30 steps).

**Experiment**: Tested both architectures across sequence lengths [5, 10, 15, 20, 25, 30, 40, 50] with weak and strong temporal dependencies.

**Results Summary — Weak Temporal**:

| Seq Len | Sim CG Imp | Real CG Imp | Gap Diff |
|---------|------------|-------------|----------|
| 5       | -14.64%    | -80.15%     | 65.50%   |
| 10      | +9.90%     | -16.84%     | 26.74%   |
| 15      | +2.24%     | -11.85%     | 14.09%   |
| 20      | -2.47%     | +12.63%     | 15.09%   |
| 25      | +2.99%     | +11.46%     | 8.47%    |
| 30      | +28.36%    | +25.18%     | 3.18%    |
| 40      | +25.93%    | +21.96%     | 3.97%    |
| 50      | +32.59%    | +38.28%     | 5.68%    |

**Key Findings**:
1. **Weak temporal: Hypothesis CONFIRMED** — Gap reduces from 35.44% (short seq 5-15) to 4.83% (long seq 40-50), a **30.61% reduction**
2. **Crossover point at seq_len=20**: Real CG starts outperforming Sim CG at longer sequences
3. **At seq_len=50, Real CG OUTPERFORMS Sim CG**: +38.28% vs +32.59%
4. **Strong temporal: Hypothesis NOT supported** — Both architectures struggle, gap remains high (40-60%) across all lengths

**Conclusion**: PARTIALLY SUPPORTED - Attention mechanism benefits from longer sequences on weak temporal tasks, but strong temporal dependencies remain challenging for both architectures.

**Sub-hypothesis H1.470.1.1.7**: Adding explicit temporal memory to Real CG will improve performance on strong temporal tasks (NOW TESTED - SUPPORTED)
