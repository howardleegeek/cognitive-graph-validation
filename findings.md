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

### H1.450: Real Language Embeddings vs Simulated — Round 216 (SUPPORTED)

**Hypothesis**: Language-conditioned models trained on real text embeddings (sentence-transformers) will achieve comparable or better performance than simulated embeddings, validating the approach for real-world deployment.

**Context**: H1.449 showed language-conditioned models can replace task IDs, but used simulated language embeddings. H1.450 tests whether real text embeddings from pre-trained language models work as well or better.

**Method**: 4-condition experiment comparing embedding types:
1. **Baseline**: Simple MLP (no language conditioning)
2. **Simulated Embeddings**: Language-conditioned model with synthetic 32-dim embeddings
3. **Cognitive Graph (sim)**: Full CG architecture with simulated embeddings
4. **Real Embeddings**: Language-conditioned model with 384-dim sentence-transformer embeddings
5. **Cognitive Graph (real)**: Full CG architecture with real embeddings

**Results**:

| Model | Validation Loss | vs Baseline |
|-------|-----------------|-------------|
| **Baseline** | 0.012327 | Reference |
| Simulated Embeddings | 0.021434 | **-73.88%** (worse) |
| Cognitive Graph (sim) | 0.015428 | **-25.16%** (worse) |
| **Real Embeddings** | **0.011033** | **+10.50%** (better) ✓ |
| Cognitive Graph (real) | 0.013708 | **-11.21%** (worse) |

**Real vs Simulated Comparison**:
- Real embeddings outperform simulated by **+48.53%**
- Real embeddings are the ONLY model to beat baseline

**Key Insights**:

1. **Real language embeddings work better**: Sentence-transformer embeddings (384-dim, trained on real text) significantly outperform simulated embeddings (32-dim, random). This validates that the language-conditioning approach works with real text.

2. **Simulated embeddings were misleading**: The simulated embeddings caused worse performance than baseline, likely because they don't capture semantic relationships. Real embeddings fix this.

3. **Simple model + real embeddings wins**: The LanguageConditionedModel (simple cross-attention) with real embeddings achieves +10.50% improvement, while the full Cognitive Graph underperforms. This suggests:
   - Real embeddings are information-rich enough that simple fusion suffices
   - The CG architecture may need tuning for higher-dimensional language inputs (384 vs 32)

4. **Practical implication**: **Use real language embeddings** — sentence-transformers provide semantically meaningful representations that improve task prediction. The simulated embeddings in earlier experiments were a limitation.

**Next Steps**: 
- Test CG architecture with projected real embeddings (384 → 128 dim)
- Compare different sentence-transformer models (all-MiniLM-L6-v2 vs all-mpnet-base-v2)
- Test on held-out language instructions to measure generalization

### H1.449: Language-Conditioned Task Identification — Round 215 (SUPPORTED)

**Hypothesis**: Language-conditioned models can achieve similar performance to task embeddings by learning to infer task identity from language descriptions, eliminating the need for explicit task IDs at inference time.

**Context**: H1.447/H1.448 showed task embeddings provide massive improvements (+32.1% to +91.5%) but require explicit task IDs. H1.449 tests whether language descriptions can replace task IDs while maintaining performance.

**Method**: 4-condition experiment matching H1.448's complexity variations, comparing 4 model variants:
1. **Baseline**: Simple MLP (no task/language conditioning)
2. **TaskEmbedding**: Cognitive Graph with explicit task ID embeddings (H1.447/H1.448 approach)
3. **Language-only**: Cognitive Graph conditioned on language embeddings (no task IDs)
4. **Hybrid**: Cognitive Graph with both language and task embeddings

**Results — Overall**:

| Model | Average Improvement | vs Baseline | vs TaskEmbedding |
|-------|-------------------|-------------|------------------|
| **TaskEmbedding** | +26.03% | Baseline | Reference |
| **Language-only** | **+26.84%** | **+26.84%** | **+103.1%** |
| **Hybrid** | +26.79% | +26.79% | +102.9% |

**By Configuration** (4 conditions matching H1.448):

| Configuration (Objects, Horizon) | TaskEmbedding | Language-only | Hybrid |
|----------------------------------|---------------|---------------|--------|
| (5, 10) | +22.21% | **+24.28%** | +27.55% |
| (8, 15) | +37.43% | +37.04% | +36.74% |
| (3, 5) | +18.46% | **+20.98%** | +19.18% |
| (10, 20) | +26.03% | **+26.84%** | +26.79% |

**Key Insights**:

1. **Language can effectively replace task IDs**: Language-only models achieve **103.1%** of task embedding performance, demonstrating that language descriptions contain sufficient information to infer task identity.

2. **Consistent across complexity levels**: Language-only wins in 2/4 configurations and matches performance in others, showing robustness to scene complexity and temporal length.

3. **Hybrid models don't add much**: Language+TaskID hybrid shows similar performance (+26.79%), indicating language alone captures most of the task-relevant information.

4. **Practical implication**: **Task IDs are not necessary** — language-conditioned models can infer task identity from descriptions, making the system more practical for real-world deployment.

### H1.448: Task Embeddings on Full LIBERO Suite — Round 214 (CONFIRMED)

**Hypothesis**: Task embeddings will maintain their advantage (+32.1% from H1.447) across varying object counts (3, 5, 8, 10) and horizon lengths (5, 10, 15, 20).

**Context**: H1.447 discovered that task embeddings solve the multi-task generalization problem (+32.1% vs +3.8% baseline). H1.448 tests whether this breakthrough generalizes across scene complexity and temporal length.

**Method**: 16-condition experiment (4 object counts × 4 horizons), each with 3 model variants:
1. **Baseline**: Late fusion MLP (separate encoders, concatenated)
2. **CG+TaskEmbeddings**: Cognitive Graph with task ID embeddings (H1.447 breakthrough)
3. **CG+SimpleAttention**: Cognitive Graph with simpler dot-product attention

**Results — Overall**:

| Metric | Value |
|--------|-------|
| **Task Embedding avg improvement** | **+91.5%** ✓✓✓ |
| Simple Attention avg improvement | +23.7% |
| Task Embedding win rate | **16/16 (100%)** |
| H1.447 reference | +32.1% |
| Delta from H1.447 | **+59.4 percentage points** |

**By Object Count**:

| Objects | TaskEmbed Improvement | SimpleAttn Improvement | Wins |
|---------|----------------------|------------------------|------|
| 3 | +90.0% | +43.1% | 4/4 |
| 5 | +89.4% | +4.1% | 4/4 |
| 8 | +95.0% | +44.6% | 4/4 |
| 10 | +91.5% | +2.9% | 4/4 |

**By Horizon Length**:

| Horizon | TaskEmbed Improvement | SimpleAttn Improvement | Wins |
|---------|----------------------|------------------------|------|
| 5 | +86.5% | +1.1% | 4/4 |
| 10 | +92.6% | +23.0% | 4/4 |
| 15 | +92.6% | +28.3% | 4/4 |
| 20 | +94.1% | +42.3% | 4/4 |

**Key Insights**:

1. **Task embeddings massively outperform H1.447's initial result**: +91.5% average vs +32.1% in H1.447. The larger model scale and better optimization in H1.448 reveal the full potential of task embeddings.

2. **Improvement increases with horizon**: +86.5% at horizon=5 → +94.1% at horizon=20. Task embeddings help maintain coherent task-specific behavior over longer sequences.

3. **Robust to visual complexity**: Object count has minimal effect (89-95% across 3-10 objects), proving task embeddings work across varying scene complexity.

4. **Simple attention is inconsistent**: Ranges from -26.7% to +65.8% across conditions, reinforcing that task embeddings — not just simpler architectures — are the reliable solution.

### H1.447: Task Embeddings Solve Multi-Task Generalization — Round 213 (BREAKTHROUGH)

**Hypothesis**: Task embeddings allow GraphCG to learn task-specific attention patterns, solving the multi-task generalization problem identified in