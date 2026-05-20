# Round 253 Summary — LSTM Dominance Ablation Study

## H1.470.1.1.14: SUPPORTED

This round investigated WHY LSTM is so dominant (84.33% avg improvement vs baseline) by ablating its components. We tested 6 architectures across 3 task types to isolate whether temporal recurrence, separated encoding, or both are the critical factors.

**Key result**: Temporal processing is the dominant factor — removing recurrence from LSTM makes it 105.79% worse than the full LSTM. However, separated encoding provides an additional 19.42% advantage over unified encoding even with the same temporal processing. Most strikingly, simple 1D convolutions with separated encoders (+91.51%) nearly match LSTM (+93.85%) with only a 2.34% gap, suggesting the specific recurrence mechanism matters less than having ANY temporal processing with separated encoders.

**Implication**: The optimal architecture for these tasks is separated encoders → temporal processing → simple fusion (concatenation). This is exactly what the V-JEPA + LLM alignment approach does — the very approach the Cognitive Graph was designed to replace. After 253 rounds of consistently contradictory evidence, the CG hypothesis should be formally abandoned.
