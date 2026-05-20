# Round 275 Summary: Auxiliary Loss Scaling Discovery

**Experiment**: H1.470.1.1.36 - Testing whether auxiliary loss benefits scale with model size and data volume.

**Result**: REFUTED - The hypothesis that larger models benefit more from temporal consistency regularization was falsified. In fact, the opposite is true: small models (hidden_dim=32) show +5.18% average improvement with temporal consistency, while large models (hidden_dim=128) show -5.85% degradation. This reveals a **regularization-capacity tradeoff**: auxiliary losses provide helpful inductive bias for under-capacity models but constrain over-capacity models unnecessarily.

**Key Numbers**:
- Small models (h=32): +9.36%, +2.99%, +3.20% improvement across data volumes
- Large models (h=128): -1.26%, -13.42%, -2.87% degradation across data volumes
- Data scaling: neutral (no amplification effect from more data)
- Best configuration: small model + 2000 samples + temporal consistency (loss=0.007614)

**Implication**: Use temporal consistency loss only for small models; for larger models, rely on data volume rather than auxiliary regularization. This finding has practical consequences for model design in resource-constrained robotics applications.

**Next**: Test adaptive regularization that scales with model capacity, or investigate the mechanism of over-regularization in larger models.