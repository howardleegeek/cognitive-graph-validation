# Round 208 Summary — Cognitive Graph Validation

**Experiment**: H1.442 - Adaptive Node GraphCG on LIBERO Tasks

**Critical Finding**: GraphCG performs **39.8% worse** than MLP on LIBERO-style manipulation tasks, contradicting the +29.1% improvement seen on synthetic tasks (H1.441). This reveals a fundamental domain transfer failure: synthetic transformation tasks do not capture the complexity of real manipulation action prediction.

**Key Numbers**:
- MLP baseline MSE: 0.1449
- GraphCG (fixed 6 nodes) MSE: 0.2027 (-39.8% vs MLP)
- GraphCG (adaptive nodes) MSE: 0.2092 (-44.4% vs MLP)
- Tested across 4 task types: simple_pick (2 obj), pick_place (3 obj), multi_object (5 obj), long_horizon (7 obj)

**Implication**: The GraphCG architectural advantage on synthetic state-prediction tasks does not transfer to action-prediction tasks. This suggests either (1) synthetic tasks have exploitable structure not present in real tasks, or (2) MLP is fundamentally better suited for action prediction. Next step: investigate task structure differences to understand the discrepancy.