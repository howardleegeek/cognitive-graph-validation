# Round 277 Summary: Architecture-Dependent Regularization Investigation

## Experiment: H1.470.1.1.38

**Objective**: Test whether the over-regularization effect observed in H1.470.1.1.36 (temporal consistency hurts large models by -5.85%) is specific to the cognitive graph architecture or occurs in simpler models too.

**Key Finding**: Both simple GRU and cognitive graph architectures show over-regularization for larger models, but at different capacity thresholds. Simple GRU shows negative effects starting at h=64 (-1.44%) worsening at h=128 (-6.17%), while cognitive graph shows strong benefits at h=128 (+11.83%) but over-regularization at h=256 (-9.38%).

**Implication**: Over-regularization is not purely architecture-dependent but depends on the ratio of model capacity to task complexity. The effect occurs when model capacity exceeds some threshold relative to the task, with simpler architectures having lower thresholds than more complex ones.

**Next Step**: Investigate task-dependent regularization scaling based on task complexity metrics to develop adaptive regularization that considers both architecture complexity and task difficulty.