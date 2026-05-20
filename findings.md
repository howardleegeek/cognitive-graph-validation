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

### H1.470.1.1.8: Hierarchical Temporal Memory — Round 247 (PARTIALLY_SUPPORTED)

**Hypothesis**: Hierarchical temporal memory (multiple LSTM layers at different timescales) will further improve performance on longer sequences compared to single-layer LSTM.

**Previous Finding (H1.470.1.1.7)**: Single LSTM provides +80.44% improvement on strong temporal tasks (seq_len 10-20).

**Experiment**: Tested three architectures on strong temporal tasks with sequence lengths 20-50:
1. Single LSTM (baseline from H1.470.1.1.7)
2. Hierarchical LSTM (2 levels: fast + slow timescales)
3. Hierarchical LSTM (3 levels: fast + medium + slow timescales)

**Results Summary**:

| Seq Len | Baseline Loss | Single LSTM | Hierarchical 2 | Hierarchical 3 |
|---------|---------------|-------------|----------------|----------------|
| 20      | 0.1648        | 0.0046 (97.19%) | 0.0046 (97.23%) | 0.0045 (97.25%) |
| 30      | 0.1645        | 0.0047 (97.17%) | 0.0046 (97.18%) | 0.0043 (97.40%) |
| 40      | 0.1575        | 0.0044 (97.21%) | 0.0045 (97.13%) | 0.0043 (97.30%) |
| 50      | 0.1470        | 0.0045 (96.91%) | 0.0050 (96.61%) | 0.0045 (96.94%) |

**Average Improvement over Baseline**:
- Single LSTM: **97.12%**
- Hierarchical 2-level: **97.04%**
- Hierarchical 3-level: **97.22%**

**Hierarchical vs Single LSTM (relative improvement)**:

| Seq Len | Hier 2 vs Single | Hier 3 vs Single |
|---------|------------------|------------------|
| 20      | +1.51%           | +2.15%           |
| 30      | +0.30%           | +8.02%           |
| 40      | -3.01%           | +3.11%           |
| 50      | -9.69%           | +1.06%           |

**Key Findings**:
1. **PARTIALLY_SUPPORTED** — Hierarchical 3-level shows marginal improvement over single LSTM (97.22% vs 97.12%)
2. **Hierarchical 2-level underperforms** at longer sequences (negative relative improvement at seq_len 40, 50)
3. **Hierarchical 3-level shows best performance at seq_len 30** (+8.02% relative improvement)
4. **No clear scaling trend** — hierarchical advantage does not consistently increase with sequence length
5. **All LSTM variants dramatically outperform baseline** — the key insight is that ANY temporal memory provides ~97% improvement

**Conclusion**: PARTIALLY_SUPPORTED - Hierarchical temporal memory provides marginal additional benefit over single LSTM, but the improvement is inconsistent across sequence lengths. The primary finding remains that explicit temporal memory (any variant) is essential for strong temporal tasks.

**Sub-hypothesis H1.470.1.1.9**: Test if the benefit of hierarchical memory emerges with even longer sequences (100+ timesteps) or more complex temporal patterns (multi-scale dependencies).

---

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

---

### H1.470.1.1.6: Attention Mechanism Sequence Length Sensitivity — Round 245 (PARTIALLY_SUPPORTED)

**Hypothesis**: Attention mechanism benefits from longer sequences on weak temporal tasks but struggles with strong temporal dependencies regardless of sequence length.

**Experiment**: Tested Sim CG and Real CG across sequence lengths 5-50 with both weak and strong temporal dependencies.

**Key Findings**:
1. **Weak temporal tasks**: Gap between Sim CG and Real CG decreases with longer sequences (35.44% → 4.83%)
2. **Strong temporal tasks**: Gap remains large across all sequence lengths (36.48% → 46.61%)
3. **Crossover at seq_len=50**: Real CG (+38.28%) outperforms Sim CG (+32.59%) on weak temporal tasks
4. **Strong temporal dependencies remain challenging** for both architectures

**Conclusion**: PARTIALLY_SUPPORTED - Attention benefits from longer sequences on weak temporal tasks, but strong temporal dependencies require explicit memory mechanisms.

---

## Summary of Key Insights

1. **Temporal memory is essential**: LSTM/GRU provides +80-97% improvement on strong temporal tasks
2. **Attention alone is insufficient**: Attention-only provides ~0% improvement on strong temporal dependencies
3. **Hierarchical memory provides marginal benefit**: 3-level hierarchy shows ~0.1% additional improvement over single LSTM
4. **Sequence length matters for weak temporal tasks**: Longer sequences help attention mechanisms
5. **Strong temporal dependencies require explicit memory**: Regardless of sequence length or architecture complexity

## Next Steps

- H1.470.1.1.9: Test hierarchical memory on very long sequences (100+ timesteps)
- H1.470.1.1.10: Investigate if hierarchical memory helps with multi-scale temporal patterns
- Consider alternative memory architectures (Transformer-XL, Compressive Transformer)