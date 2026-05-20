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

### H1.470.1.1.15: Late-Fusion Architecture Test — Round 254 (SUPPORTED)

**Hypothesis**: Based on H1.470.1.1.14 findings, the optimal architecture should be: separated encoders → temporal processing → late concatenation. Late fusion preserves the benefits of separated encoding (no cross-modal interference) while adding temporal processing to each modality independently.

**Prediction**: Late-fusion architectures will outperform early fusion on crossmodal tasks while matching on temporal tasks.

**Experiment**: Tested 6 architectures across 3 task types:
1. Baseline: separate encoders → concat → output (no temporal)
2. LSTM-early: separate encoders → concat → LSTM → output (early fusion)
3. LSTM-late: separate encoders → LSTM each → concat → output (late fusion)
4. TempConv-early: separate encoders → concat → 1D conv → output
5. TempConv-late: separate encoders → 1D conv each → concat → output
6. Cognitive Graph: unified encoder → GNN → output (reference)

**Results Summary** (improvement vs baseline):

| Architecture | Temporal-Only | Crossmodal-Only | Combined |
|-------------|---------------|-----------------|----------|
| Baseline | 0.00% | 0.00% | 0.00% |
| LSTM-early | +94.24% | +2.50% | +77.13% |
| **LSTM-late** | **+95.76%** | **+65.43%** | **+79.90%** |
| TempConv-early | +86.46% | +1.07% | +75.40% |
| TempConv-late | +95.59% | +12.80% | +80.83% |
| Cognitive Graph | -11.14% | -8.04% | -19.15% |

**Late vs Early Fusion Comparison**:

| Task | LSTM Late-Early | TempConv Late-Early |
|------|-----------------|---------------------|
| Temporal-Only | +1.52% | +9.13% |
| Crossmodal-Only | **+62.92%** | **+11.73%** |
| Combined | +2.76% | +5.43% |

**Key Findings**:
1. **Late fusion dramatically improves crossmodal performance**: LSTM-late achieves +65.43% vs +2.50% for LSTM-early on crossmodal tasks (62.92% improvement gap)
2. **Late fusion maintains temporal performance**: LSTM-late matches LSTM-early on temporal tasks (95.76% vs 94.24%)
3. **Cognitive Graph consistently underperforms**: -11.14% to -19.15% vs baseline across all tasks
4. **Best overall architecture**: LSTM-late or TempConv-late, both achieving ~80% improvement on combined tasks
5. **Critical insight**: Processing each modality's temporal dynamics independently before fusion is superior to early fusion

**Conclusion**: SUPPORTED. Late-fusion architecture (separate temporal processing per modality) significantly outperforms early fusion, especially on crossmodal tasks. This provides a clear architectural direction: maintain modality separation through temporal processing, fuse only at the final stage.

---

### H1.470.1.1.14: LSTM Dominance Ablation — Round 253 (SUPPORTED)

**Hypothesis**: LSTM's dominance comes from its combination of (a) separated modality encoding and (b) temporal recurrence processing.

**Key Results**:
- Temporal processing is the DOMINANT factor: LSTM (+93.85%) vs LSTM-FeedForward (-11.93%) = 105.79% gap
- Separated+Temporal ≈ LSTM: only 2.34% gap between 1D convolutions and LSTM
- Unified encoding underperforms separated encoding by 19.42% even with same temporal processing
- Baseline wins on crossmodal and combined tasks when no temporal processing needed

**Conclusion**: SUPPORTED. LSTM's dominance comes from temporal recurrence, not unified representation. The optimal architecture is: separated encoders → temporal processing → simple fusion (concatenation).

---

### H1.470.1.1.12: Hybrid LSTM + Cognitive Graph Architecture — Round 251 (REFUTED)

**Hypothesis**: Combining LSTM (optimal for temporal processing) with cognitive graph cross-modal attention (optimal for physical-semantic fusion) provides synergistic benefits.

**Key Results**:
- No synergy found: hybrid architectures don't outperform standalone LSTM
- CG cross-modal attention consistently degrades performance
- Baseline wins on crossmodal tasks

**Conclusion**: REFUTED. CG cross-modal attention provides no benefit over simple concatenation.

---

## Overall Research Direction

After 254 rounds of experimentation, the evidence strongly contradicts the original Cognitive Graph hypothesis:

1. **Unified representation is harmful**: Separated encoding consistently outperforms unified encoding by 19-62%
2. **Cross-modal attention is counterproductive**: Simple concatenation outperforms attention-based fusion
3. **Temporal processing is critical**: LSTM/1D-conv provides 90%+ improvement on temporal tasks
4. **Late fusion is optimal**: Processing modalities independently before final fusion achieves best results

**Recommended Architecture** (based on experimental evidence):
```
Observation → Encoder → Temporal Processor (LSTM/1D-Conv) ─┐
                                                            ├→ Concat → Action Head
Language → Encoder → Temporal Processor (LSTM/1D-Conv) ────┘
```

This architecture:
- Maintains modality separation through encoding and temporal processing
- Fuses only at the final stage (late fusion)
- Achieves 80%+ improvement over baseline on combined tasks
- Outperforms Cognitive Graph by 60-100% across all task types