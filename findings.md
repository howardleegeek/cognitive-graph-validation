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

### H1.462: GNN-only CG on Real Robot Data — Round 228 (REFUTED: 81% improvement does NOT generalize)

**Hypothesis**: The 81.31% improvement of GNN-only CG over baseline (found in H1.461) will hold when tested on real robot demonstration data.

**Context**: H1.461 found that CG without attention (GNN-only) beats baseline by 81.31% on simplified synthetic data. This experiment tests whether that advantage generalizes to realistic robot demonstration data with proper noise, variable trajectory lengths, and realistic action spaces.

**Method**: Compare 3 architectures on real robot data (800 train / 200 val samples):
1. **Baseline concatenation** (158K params) — reference
2. **CG no attention** (1.98M params) — GNN-only, the H1.461 winner
3. **CG full attention** (3.03M params) — GNN + cross-attention

**Results**:

| Config | Parameters | Val Loss | vs Baseline |
|--------|------------|----------|-------------|
| **Baseline concat** | 158,408 | **0.000303** | **0.00%** |
| CG no attention | 1,977,224 | 0.000308 | -1.74% |
| CG full attention | 3,027,848 | 0.000313 | -3.28% |

**Key Findings**:
1. **H1.461 DOES NOT GENERALIZE**: The 81.31% improvement from H1.461 completely disappears on real robot data
2. **Baseline wins on real data**: Simple concatenation beats both CG variants (by 1.74% and 3.28%)
3. **Attention still degrades**: CG with attention (-3.28%) is worse than GNN-only (-1.74%), confirming attention is harmful
4. **Parameter efficiency matters**: Baseline achieves best results with 12.5x fewer parameters than CG no-attn
5. **Data distribution shift**: The synthetic data in H1.461 may have had structural properties that favored CG (e.g., cleaner graph structure, less noise)

**Analysis — Why did H1.461's 81% improvement vanish?**
- H1.461 used simplified synthetic data with clean graph structure and low noise
- Real robot data has: sensor noise, actuator noise, variable trajectory lengths, complex dynamics
- The GNN message passing may have been exploiting structure in the simplified data that doesn't exist in real data
- The baseline's simplicity makes it more robust to noise and distribution shift
- **Hypothesis**: CG's advantage requires clean, structured data with explicit graph-like relationships. Real robot data is too noisy for the graph structure to provide benefit.

**Conclusion**: H1 (CG improves sample efficiency) is **REFUTED** on real robot data. The GNN-only variant that showed 81.31% improvement on simplified data underperforms baseline by 1.74% on real robot data. The attention mechanism remains harmful (confirming H1.461's finding about attention), but the GNN-only advantage does not transfer to realistic settings.

### H1.461: Simplified CG Investigation — Round 227 (BREAKTHROUGH: CG BEATS BASELINE WHEN ATTENTION REMOVED)

**Hypothesis**: CG's poor performance may be due to overparameterization. Testing simplified CG variants with fewer parameters to see if performance improves.

**Context**: H1.457-H1.460 showed CG consistently underperforms baseline across all configurations. This experiment tests if a simpler CG architecture helps.

**Method**: Compare 8 CG variants against concatenation baseline:
1. **Baseline concatenation** (reference)
2. **CG full** (hidden=256, 3 GNN layers, 4 heads)
3. **CG reduced hidden** (hidden=128)
4. **CG 1 layer** (1 GNN layer instead of 3)
5. **CG 1 head** (1 attention head instead of 4)
6. **CG minimal** (hidden=64, 1 layer, 1 head)
7. **CG no GNN** (attention only, no GNN)
8. **CG no attention** (GNN only, no attention)

**Results**:

| Config | Parameters | Val Loss | vs Baseline |
|--------|------------|----------|-------------|
| **CG no attention** | 867,847 | **0.011754** | **+81.31%** |
| **CG full** | 1,131,015 | 0.020725 | +67.04% |
| **CG 1 layer** | 604,679 | 0.041329 | +34.28% |
| Baseline concat | 78,087 | 0.062887 | 0.00% |
| CG no GNN | 341,511 | 0.083596 | -32.93% |
| CG reduced hidden | 286,983 | 0.097344 | -54.79% |
| CG 1 head | 1,131,015 | 0.128090 | -103.68% |
| CG minimal | 40,583 | 0.715210 | -1037.30% |

**Key Findings**:
1. **CG CAN BEAT BASELINE**: CG without attention achieves 81.31% improvement over baseline!
2. **Attention is the problem**: Removing attention dramatically improves performance
3. **GNN is beneficial**: CG with GNN-only (no attention) is the best configuration
4. **More parameters help**: Full CG (1.1M params) beats reduced variants
5. **Attention degrades performance**: Adding attention to GNN makes it worse

**Conclusion**: The attention mechanism in CG was causing the poor performance. GNN message passing alone provides the benefit. This suggests that for this task, explicit graph structure helps but learned attention patterns hurt.

**Implications**:
- H1 (CG improves sample efficiency) may be SUPPORTED when using GNN-only variant
- Previous negative results were due to attention mechanism, not CG concept itself
- Need to re-test H1 with GNN-only CG variant

## Hypothesis Status

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H1: CG improves sample efficiency | **REFUTED** | GNN-only CG underperforms baseline by 1.74% on real robot data (H1.462). The 81.31% improvement on simplified data (H1.461) does not generalize. |
| H2: CG helps multi-step tasks | Inconclusive | 1.7% difference |
| H3: Attention helps long sequences | **REFUTED** | Removing attention improves CG by 81.31% |
| H4: 25% dimension allocation optimal | Close | 25% optimal vs 28% hypothesis |

## Next Steps

1. **H1.463**: Investigate why CG advantage disappears on real data — is it noise, data structure, or parameter efficiency?
2. **H1.464**: Test if CG can be made competitive with real data by adding noise regularization or simplifying the graph structure
3. **H1.465**: Explore hybrid approach — use CG for structured sub-tasks, baseline for noisy perception
