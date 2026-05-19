# Round 195 Summary: Temporal Sequence Modeling for Per-Object CG

**Experiment**: H1.429 - Testing whether adding LSTM/GRU temporal modeling to Per-Object CG improves multi-step task performance.

**Key Results**:
- **GRU helps multi-stage tasks**: Per-Object CG + GRU achieves -7.5% MSE vs baseline on multi-stage tasks (best result), compared to -4.8% for vanilla Per-Object CG
- **LSTM fails**: LSTM degrades both spatial (+115% worse) and multi-stage (+17% worse) performance significantly
- **Modest improvement**: GRU provides only +2.9% improvement over vanilla CG on multi-stage tasks

**Conclusion**: PARTIALLY_SUPPORTED. Temporal modeling with GRU helps multi-stage tasks, but the effect is smaller than expected. LSTM fails completely. The hypothesis that temporal dependencies are the missing piece for multi-stage tasks is partially confirmed, but Per-Object CG already captures some temporal structure through object state evolution.

**Next Step**: H1.430 will test attention-based temporal aggregation (Transformer) vs RNN-based (GRU) to see if attention mechanisms work better for temporal dependencies in multi-stage tasks.