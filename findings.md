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

### H1.470.1.1.12: Hybrid LSTM + Cognitive Graph Architecture — Round 251 (REFUTED)

**Hypothesis**: Combining LSTM (optimal for temporal processing) with cognitive graph cross-modal attention (optimal for physical-semantic fusion) provides synergistic benefits that neither architecture achieves alone.

**Prediction**: Hybrid LSTM+CG outperforms both standalone LSTM and standalone CG by >5% on tasks requiring BOTH temporal reasoning AND cross-modal grounding.

**Experiment**: Tested 5 architectures across 3 task types (temporal-only, cross-modal-only, combined):
1. Baseline (separate encoders + concatenation)
2. Standard LSTM (temporal processing only)
3. Cognitive Graph (cross-modal attention only)
4. Hybrid LSTM+CG: CG fusion at each timestep → LSTM temporal processing
5. Hybrid CG+LSTM: CG fusion on sequence mean → LSTM with CG context

**Results Summary**:

| Architecture | Params | Temporal-Only | Cross-Modal-Only | Combined |
|-------------|--------|---------------|-------------------|----------|
| Baseline | 61K | 0.3026 | 0.3879 | 0.3783 |
| LSTM | 358K | 0.1175 (+61.16%) | 0.4480 (-15.47%) | 0.1727 (+54.36%) |
| CG | 1,995K | 0.3033 (-0.24%) | 0.8029 (-106.97%) | 0.3852 (-1.81%) |
| Hybrid LSTM+CG | 1,462K | 0.1631 (+46.12%) | 0.8290 (-113.70%) | 0.1514 (+59.99%) |
| Hybrid CG+LSTM | 1,537K | 0.1141 (+62.29%) | 0.5702 (-46.97%) | 0.1419 (+62.49%) |

**Synergy Analysis** (hybrid improvement vs best single architecture):

| Task | Best Single | Hybrid LSTM+CG | Synergy? |
|------|-------------|----------------|----------|
| Temporal-Only | LSTM +61.16% | Hybrid CG+LSTM +62.29% | NO (+1.13%) |
| Cross-Modal-Only | Baseline (all worse) | All negative | NO |
| Combined | LSTM +54.36% | Hybrid CG+LSTM +62.49% | YES (+8.13%) |

**Key Findings**:
1. **Hypothesis REFUTED** — Hybrid does NOT show consistent synergistic benefits across tasks
2. **Only 1/3 tasks show synergy**: Combined task shows +8.13% improvement over best single (LSTM)
3. **Average synergy: -35.88%** — hybrids are worse than best single on average
4. **LSTM dominates temporal tasks**: +61.16% on temporal-only, +54.36% on combined
5. **CG performs poorly on all tasks**: Never beats baseline, even on cross-modal-only (-106.97%)
6. **Hybrid CG+LSTM is best hybrid**: Slightly better than Hybrid LSTM+CG on all tasks, suggesting CG-as-context (front-end) works better than CG-per-timestep (in-loop)
7. **Parameter efficiency concern**: Hybrids use 4-25x more parameters than LSTM for marginal gains

**Conclusion**: REFUTED — The hybrid approach does not provide consistent synergistic benefits. LSTM alone remains the most efficient and effective architecture. The CG component adds significant parameter overhead (1.4M+ params) without proportional performance gains. The one exception (combined task, +8.13%) is insufficient to justify the architectural complexity.

---

### H1.470.1.1.11: LSTM Architectural Improvements — Round 250 (REFUTED)

**Hypothesis**: LSTM performance can be further improved with better initialization, regularization, or architectural modifications (peephole connections, attention-augmented LSTM, variational LSTM).

**Previous Finding (H1.470.1.1.10)**: Single LSTM remains optimal for strong temporal dependencies, outperforming all alternatives (Transformer-XL, SWA, Global Attention) by 57-223%.

**Experiment**: Tested five LSTM variants on strong temporal tasks with sequence lengths 60 and 100:
1. Standard LSTM (baseline comparison)
2. Peephole LSTM (gated access to cell state)
3. Zoneout LSTM (regularization via zoneout)
4. Attention-augmented LSTM (attention over hidden states)
5. Variational LSTM (variational inference over weights)

**Results Summary**:

| Architecture | Seq 60 | Seq 100 | Avg Improvement |
|-------------|--------|---------|-----------------|
| Standard LSTM | +0.92% | +0.92% | +0.92% |
| Peephole LSTM | +0.32% | +0.32% | +0.32% |
| Zoneout LSTM | +0.88% | +0.88% | +0.88% |
| Attention LSTM | -0.63% | -0.63% | -0.63% |
| Variational LSTM | +0.50% | +0.50% | +0.50% |

**Key Findings**:
1. **No variant provides >5% improvement** over standard LSTM
2. **Zoneout performs nearly identically**: -0.04% vs standard LSTM
3. **Attention-augmented performs WORST**: -1.55% vs standard LSTM
4. **Standard LSTM remains optimal** — already well-optimized

**Conclusion**: REFUTED — Standard LSTM is already well-optimized. No architectural modification provides meaningful improvement.

---

### H1.470.1.1.10: Alternative Memory Architectures for Very Long Sequences — Round 249 (REFUTED)

**Hypothesis**: Alternative memory architectures (Transformer-XL style recurrence, sliding window attention, global attention) may better handle very long sequences where hierarchical LSTM advantage decreases.

**Previous Finding (H1.470.1.1.9)**: 
- Hierarchical 3-level shows consistent improvement over single LSTM (77.63% vs 73.53%)
- BUT advantage DECREASES with sequence length (correlation -0.984)
- At seq_len=150: Hier3 vs Single only +11.38% (vs +21.24% at seq_len=60)

**Experiment**: Tested four architectures on strong temporal tasks with sequence lengths 60-200:
1. Single LSTM (baseline comparison)
2. Transformer-XL Memory (segment-level recurrence)
3. Sliding Window Attention (local window + external memory bank)
4. Global Attention (full sequence self-attention)

**Results Summary**:

| Seq Len | Baseline Loss | Single LSTM | Transformer-XL | SWA | Global Attn |
|---------|---------------|-------------|----------------|-----|-------------|
| 60      | 0.0712        | 0.0199 (72.0%) | 0.0468 (34.4%) | 0.0538 (24.5%) | 0.0644 (9.6%) |
| 100     | 0.0553        | 0.0192 (65.3%) | 0.0345 (37.7%) | 0.0445 (19.6%) | 0.0521 (5.8%) |
| 150     | 0.0470        | 0.0179 (62.0%) | 0.0293 (37.6%) | 0.0391 (16.7%) | 0.0454 (3.5%) |
| 200     | 0.0411        | 0.0156 (62.1%) | 0.0246 (40.1%) | 0.0345 (16.1%) | 0.0399 (2.8%) |

**Average Improvement over Baseline**:
- Single LSTM: **65.35%**
- Transformer-XL: **37.46%**
- Sliding Window Attention: **19.48%**
- Global Attention: **5.29%**

**Alternative vs Single LSTM (relative performance)**:

| Seq Len | TXL vs LSTM | SWA vs LSTM | GA vs LSTM |
|---------|-------------|-------------|------------|
| 60      | -84.1% | -135.5% | -176.2% |
| 100     | -79.2% | -131.8% | -171.3% |
| 150     | -63.7% | -117.9% | -153.7% |
| 200     | -57.7% | -121.2% | -155.8% |

**Key Findings**:
1. **Single LSTM remains optimal** across all sequence lengths
2. **All alternatives show positive scaling correlation** (improve relative to LSTM at longer sequences)
3. **TXL scaling correlation: 0.885** — improves most at longer sequences
4. **SWA scaling correlation: 0.843**
5. **GA scaling correlation: 0.848**
6. **Sequential processing outperforms parallel/segmented approaches** for strong temporal dependencies

**Conclusion**: REFUTED — Single LSTM remains optimal. All alternatives show positive scaling correlation but never surpass it.

---

## Summary of Key Insights

1. **Temporal memory is essential**: LSTM/GRU provides +65-80% improvement on strong temporal tasks
2. **Attention alone is insufficient**: Attention-only provides ~0-5% improvement on strong temporal dependencies
3. **Single LSTM is optimal**: Outperforms all alternatives (Transformer-XL, SWA, Global Attention) by 57-223%
4. **Hierarchical memory provides marginal benefit**: 3-level hierarchy shows +4.1% avg improvement over single LSTM, but advantage decreases with sequence length
5. **Alternative architectures show positive scaling but never surpass LSTM**: Transformer-XL, SWA, and Global Attention all improve relative to LSTM at longer sequences (correlation 0.84-0.89), but remain significantly worse
6. **Sequential processing is optimal for strong temporal dependencies**: LSTM's sequential nature outperforms all parallel/segmented approaches
7. **LSTM architectural modifications don't help**: Peephole, zoneout, attention-augmented, and variational LSTM all fail to improve >5% over standard LSTM
8. **Hybrid LSTM+CG does NOT provide consistent synergy**: Only 1/3 tasks show synergy (+8.13% on combined task). Average synergy: -35.88%. CG adds 1.4M+ parameters without proportional gains
9. **CG alone performs poorly**: Never beats baseline across any task type, even on cross-modal-only tasks (-106.97%)
10. **LSTM dominates**: Best single architecture on 2/3 tasks, and the hybrid that works best (CG+LSTM) is essentially LSTM with CG as a context provider

## Next Steps

- **H1.470.1.1.13**: Investigate why CG underperforms — is it the dimension mismatch (144+368=512 vs LSTM's 128), the attention mechanism, or the GNN layers?
- **H1.470.1.1.14**: Test lightweight CG variants with reduced dimensions to match LSTM parameter budget
- **H1.470.1.1.15**: Explore whether CG benefits emerge only with real robot data (vs synthetic)
- Consider whether the cognitive graph approach needs fundamentally different inductive biases for temporal tasks
