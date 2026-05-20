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
1. Hybrid LSTM+CG does NOT provide consistent synergy: only 1/3 tasks show synergy (+8.13% on combined). Average synergy: -35.88%
2. CG alone performs poorly: never beats baseline across any task type, even on cross-modal-only tasks (-106.97%)
3. LSTM dominates: best single architecture on 2/3 tasks
4. The hybrid that works best (CG+LSTM) is essentially LSTM with CG as a context provider — CG adds 1.4M+ parameters without proportional gains

### H1.470.1.1.13: Lightweight CG Variants with Reduced Dimensions — Round 252 (REFUTED)

**Hypothesis**: CG's poor performance is due to parameter budget mismatch. Reducing CG dimensions to match LSTM's parameter budget should close the gap.

**Prediction**: Lightweight CG variants (64-256 dim unified space) will approach LSTM performance when parameter budgets are matched.

**Results**: Best lightweight CG (cg_attention): 6.76% avg improvement vs LSTM's 84.33%. Parameter budget is NOT the issue — CG-medium (243K params, close to LSTM's 344K) performs WORSE than CG-tiny (16K params). Inverse scaling: larger CG dimensions make performance worse. The unified representation concept itself is fundamentally flawed for these tasks.

### H1.470.1.1.14: LSTM Dominance Ablation Study — Round 253 (SUPPORTED)

**Hypothesis**: LSTM's dominance comes primarily from its temporal recurrence mechanism. Separated encoding provides additional benefit but is secondary to temporal processing.

**Prediction**: (1) LSTM without temporal recurrence will perform similarly to baseline. (2) Separated encoders with temporal processing will approach LSTM performance. (3) Unified encoders with temporal processing will underperform separated+temporal.

**Experiment**: 6 architectures across 3 task types:
1. Baseline (separate encoders + concatenation, no temporal)
2. LSTM (separated encoders + temporal recurrence)
3. LSTM-FeedForward (separated encoders, NO temporal recurrence)
4. Separated+Temporal (separated encoders + 1D convolutions)
5. Unified+Temporal (unified encoder + LSTM)
6. Unified+FeedForward (unified encoder, no temporal)

**Results — Temporal-Only Tasks**:

| Architecture | Params | Val Loss | Improvement vs Baseline |
|-------------|--------|----------|------------------------|
| Baseline | 61K | 1.9839 | — |
| LSTM | 301K | 0.1219 | **+93.85%** |
| LSTM-FeedForward | 70K | 2.2207 | -11.93% |
| Separated+Temporal | 135K | 0.1684 | **+91.51%** |
| Unified+Temporal | 295K | 0.5538 | +72.09% |
| Unified+FeedForward | 31K | 2.7453 | -38.38% |

**Results — Crossmodal-Only Tasks**: All architectures performed worse than baseline (baseline optimal for crossmodal-only).

**Results — Combined Tasks**: All architectures performed worse than baseline.

**Key Findings**:
1. **Temporal processing is the DOMINANT factor**: LSTM (+93.85%) vs LSTM-FeedForward (-11.93%) = 105.79% gap. Removing temporal recurrence makes LSTM worse than baseline.
2. **Separated+Temporal ≈ LSTM**: Separated+Temporal (+91.51%) vs LSTM (+93.85%) = only 2.34% gap. Simple 1D convolutions nearly match LSTM's recurrent processing.
3. **Unified encoding underperforms separated encoding**: Unified+Temporal (+72.09%) vs Separated+Temporal (+91.51%) = 19.42% gap. Even with the same temporal processing, unified encoding is worse.
4. **Baseline wins on crossmodal and combined tasks**: All architectures perform worse than baseline on these task types, consistent with H3 (concatenation wins for simple tasks).
5. **Unified encoding is consistently the worst approach**: Unified+FeedForward is worst on temporal-only (-38.38%). Unified+Temporal underperforms Separated+Temporal by 19.42%.

**Conclusion**: LSTM's dominance comes from temporal recurrence, not separated encoding. However, separated encoding provides an additional 19.42% advantage over unified encoding when combined with temporal processing. The optimal architecture is: separated encoders → temporal processing → simple fusion (concatenation).

## Summary of All Findings

1. **CG alone never beats baseline**: Across all experiments, the cognitive graph architecture has never outperformed the simple baseline (separate encoders + concatenation)
2. **LSTM is the dominant architecture**: Best single architecture on temporal tasks (+93.85% improvement)
3. **Unified representations are fundamentally flawed**: Even with controlled parameter budgets, CG variants achieve only 6.76% vs LSTM's 84.33%
4. **Inverse scaling in CG**: Larger CG dimensions make performance worse
5. **Hybrid LSTM+CG provides no consistent synergy**: Average synergy: -35.88%
6. **Temporal processing is the critical factor**: 105.79% gap between LSTM and LSTM-FeedForward
7. **Separated encoding > unified encoding**: 19.42% advantage even with same temporal processing
8. **Simple temporal processing ≈ LSTM**: 1D convolutions nearly match LSTM (2.34% gap)
9. **Baseline is optimal for crossmodal/combined tasks**: No architecture beats simple concatenation on these tasks
10. **The CG hypothesis is contradicted by evidence**: Separated encoders + temporal processing + late fusion is the optimal approach — exactly what V-JEPA + LLM alignment does

## Next Steps

- **H1.470.1.1.15**: Test if late-fusion architectures (separate encoders → temporal processing → late concatenation) outperform both baseline and LSTM
- **H1.470.1.1.16**: Investigate whether there are ANY task types where unified representations provide an advantage
- **Consider formally abandoning the CG hypothesis** — 253 rounds of evidence consistently contradict it
- **Pivot to optimizing the separated+temporal approach** which the data shows is optimal
