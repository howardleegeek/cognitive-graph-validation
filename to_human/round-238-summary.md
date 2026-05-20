# Round 238 Summary: H1.470.1.1 Fine-grained dimension sweep

## Experiment Executed
**H1.470.1.1**: Fine-grained dimension sweep around 768 [640, 704, 768, 832, 896] to test the hypothesis that 768 is the optimal representation dimension for Cognitive Graph on multi-step tasks.

## Key Results
- **Hypothesis REFUTED**: 768 is NOT the optimal dimension
- **New optimal point**: 832 dimensions shows best multi-step performance (+41.49% improvement)
- **768 performance**: Only +31.06% multi-step improvement (3rd best out of 5)
- **Negative gap at optimal**: Both 768 (-1.19%) and 832 (-1.76%) show CG performing BETTER on multi-step than single-step
- **High variance at 768**: Standard deviation 8.79-9.36% suggests instability
- **Clear peak**: Performance degrades at 896 (+28.88%), confirming non-monotonic relationship

## Implications
The optimal representation dimension for Cognitive Graph appears to be ~832, not 768. At this optimal dimension, CG actually performs better on multi-step tasks than single-step tasks (negative improvement gap). This suggests that with the right representation size, the unified architecture can handle temporal dependencies more effectively than simpler tasks.

## Next Step
H1.470.1.1.1: Even finer sweep around 832 [800, 816, 832, 848, 864] to confirm the exact optimal dimension.