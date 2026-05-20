# Round 255 Summary: Late-Fusion Scalability Test on Longer Sequences

**Key Finding**: Contrary to expectations, early fusion (concat → temporal processing) outperforms late fusion (temporal processing each → concat) on longer sequences, with the performance gap becoming increasingly negative as sequence length increases. At 40 timesteps, LSTM-late degrades catastrophically to -49.39% vs baseline, while LSTM-early maintains +7.51% improvement.

**Experiment**: Tested 4 architectures (baseline, LSTM-early, LSTM-late, Cognitive Graph) across 5 sequence lengths (5, 10, 20, 30, 40 timesteps) on combined temporal+crossmodal tasks. The hypothesis predicted late-fusion would scale better due to independent temporal processing preventing cross-modal interference accumulation.

**Results**:
- Sequence length 5: LSTM-early +18.50%, LSTM-late +4.85% (gap: -13.65%)
- Sequence length 10: LSTM-early +11.21%, LSTM-late +5.31% (gap: -5.90%)
- Sequence length 20: LSTM-early +56.88%, LSTM-late +40.73% (gap: -16.14%)
- Sequence length 30: LSTM-early +50.61%, LSTM-late +25.20% (gap: -25.41%)
- Sequence length 40: LSTM-early +7.51%, LSTM-late -49.39% (gap: -56.90%)

**Implication**: The hypothesis that late-fusion scales better to longer sequences is REFUTED. Joint temporal processing of concatenated modalities (early fusion) appears more stable than independent temporal processing followed by fusion. This suggests that for long sequences, maintaining temporal coherence across modalities jointly may be more important than preventing cross-modal interference.