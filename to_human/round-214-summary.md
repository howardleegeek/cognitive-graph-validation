# Round 214 Summary — H1.448: Task Embeddings Generalize Across Complexity

**Experiment**: H1.448 — Tested task embeddings on the full LIBERO suite across 16 conditions (4 object counts × 4 horizon lengths), comparing three model variants: baseline MLP, CG+TaskEmbeddings, and CG+SimpleAttention.

**Result**: **CONFIRMED** — Task embeddings deliver a **+91.5% average improvement** over the baseline across all 16 conditions with a **100% win rate** (16/16). This dramatically exceeds H1.447's initial +32.1% finding, suggesting the scaled-down architecture allows task embeddings to be even more effective. The improvement actually *increases* with horizon length (+86.5% at horizon=5 → +94.1% at horizon=20), proving task embeddings help maintain coherent task-specific behavior over longer sequences. Object count has minimal effect (89-95% across 3-10 objects), confirming robustness to visual complexity. Simple attention was inconsistent (-26.7% to +65.8%), reinforcing that task embeddings — not just simpler architectures — are the reliable solution.

**Next**: H1.449 will test whether the model can infer task identity from language alone (removing explicit task IDs), which would eliminate the need for task labels at inference time.
