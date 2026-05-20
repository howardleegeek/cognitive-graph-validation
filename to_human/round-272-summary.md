# Round 272 Summary — Curriculum Learning on Multi-Step Tasks (REFUTED)

## Experiment: H1.470.1.1.33

Following the REFUTED result of H1.470.1.1.32 (adaptive curriculum -17.16% worse than fixed on smooth trajectories), we tested whether curriculum learning would help on genuinely complex multi-step manipulation tasks (pick→place→return chains with 1-4 sub-steps).

**Result: REFUTED (-51.47% worse than baseline)**

Fixed curriculum performed -51.47% worse than baseline training on all data simultaneously. This degradation occurred across ALL complexity levels — even 1-step tasks suffered -1724% worse performance under curriculum training. Adaptive curriculum nearly matched baseline (-4.61%) but provided no improvement. Reverse curriculum and curriculum+attention performed worst (-54.56%, -55.25%).

**Key Finding**: Curriculum learning causes catastrophic forgetting between stages. Training on easy tasks first actually harms the model's ability to handle those same tasks later. Joint training on all complexities simultaneously allows the model to learn shared representations that generalize across task difficulty.

**Pattern Across 3 Experiments**:
- H1.470.1.1.31: Curriculum +81.09% on smooth trajectories (SUPPORTED)
- H1.470.1.1.32: Adaptive curriculum -17.16% on smooth trajectories (REFUTED)
- H1.470.1.1.33: Fixed curriculum -51.47% on multi-step tasks (REFUTED)

Curriculum learning only helps on very simple, homogeneous tasks. As soon as tasks involve discrete sub-goals or heterogeneous complexity, it becomes harmful.

**Next**: H1.470.1.1.34 — Test auxiliary loss approaches (sub-goal prediction, consistency losses) as an alternative to curriculum learning for multi-step tasks.
