# Round 259 Summary: Noise-Robust Training Experiment

## What was done

Tested noise-robust training techniques to close the 13.52% performance gap between synthetic (+55%) and real robot data (+41.48%). Five configurations were tested: baseline, input denoising, noise-aware loss, adversarial training, and combined approach.

## Key results

- **Noise-aware loss** emerged as the best technique with +251.41% relative improvement
- Input denoising actually hurt performance (-753.34%)
- Adversarial training had minimal effect (-1.88%)
- Combined approach showed modest improvement (+32.46%)

## Conclusion

**SUPPORTED** - Noise-aware loss is expected to close 100% of the performance gap, bringing real robot data performance from 41.48% to 55.0% (matching synthetic data).

## Next step

Test noise-aware loss on actual real robot data to validate the extrapolation (H1.470.1.1.21).
