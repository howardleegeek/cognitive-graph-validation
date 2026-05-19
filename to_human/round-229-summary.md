# Round 229 Summary

## What was done

Investigated why the 81.31% CG improvement from H1.461 completely vanished on real robot data (H1.462). Tested hypothesis that noise sensitivity is the root cause by adding increasing noise levels to synthetic data.

## Key result

**CONFIRMED: CG advantage is highly noise-sensitive.** Even tiny noise (1% of signal) destroys CG's 28% advantage on clean data. At noise=0.01, CG performs 867% WORSE than baseline. This explains why H1.461's 81% improvement doesn't generalize to real robot data — real data has inherent sensor noise, measurement errors, and distribution shift that CG cannot handle.

## Implications

- H1 (CG advantage) is effectively REFUTED for real-world deployment
- Simple concatenation baseline is more robust to noise and 12.5x more parameter-efficient
- CG only works in clean, controlled synthetic environments

## Next step

H1.464: Test if noise-robust training (data augmentation, regularization) can restore CG performance on noisy data. Or pivot to H3 (attention on longer sequences).
