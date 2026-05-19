# Round 216 Summary

**Experiment**: H1.450 — Real Language Embeddings vs Simulated

**Result**: SUPPORTED. Real sentence-transformer embeddings (384-dim, from all-MiniLM-L6-v2) significantly outperform simulated embeddings (32-dim) by **+48.53%**, achieving **+10.50%** improvement over baseline. This is the first model to beat baseline in this series, validating that language-conditioning works with real text.

**Key Numbers**:
- Baseline loss: 0.012327
- Real embeddings loss: 0.011033 (+10.50% better)
- Simulated embeddings loss: 0.021434 (-73.88% worse)
- Cognitive Graph (real) loss: 0.013708 (-11.21% worse)

**Insight**: The simple LanguageConditionedModel (cross-attention) with real embeddings beats the full Cognitive Graph architecture. This suggests the CG architecture needs tuning for higher-dimensional language inputs (384 vs 32), or that real embeddings are already information-rich enough that simple fusion suffices.

**Next**: H1.451 will test CG with projected real embeddings (384 → 128 dim) to see if the architecture can match the simple model's performance.