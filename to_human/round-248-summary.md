# Round 248 Summary - H1.470.1.1.9: Hierarchical Temporal Memory on Very Long Sequences

## Experiment Overview
Tested hierarchical temporal memory (2-level and 3-level LSTM) on very long sequences (60-150 timesteps) to see if hierarchical benefits emerge with longer sequences and multi-scale temporal patterns.

## Key Results
- **Hierarchical 3-level consistently outperforms single LSTM**: 77.63% vs 73.53% average improvement over baseline
- **BUT hierarchical advantage DECREASES with sequence length**: Strong negative correlation (-0.984) between sequence length and hierarchical benefit
- **All models degrade on very long sequences**: Performance drops from ~88% at 60 timesteps to ~65% at 150 timesteps
- **Hierarchical 2-level shows intermediate performance**: 76.45% average improvement

## Key Insight
Contrary to the hypothesis, hierarchical LSTM advantage does NOT increase with sequence length. While hierarchical memory provides consistent improvement over single LSTM, all LSTM-based architectures struggle with very long sequences (150+ timesteps), suggesting LSTM may not be optimal for extremely long temporal dependencies.

## Next Step
Investigate alternative memory architectures (Transformer-XL, state space models) for very long sequences, or test hierarchical memory with different multi-scale patterns.