# Round 230 Summary: Noise-Robust Training for Cognitive Graph

**Experiment**: H1.464 tested whether noise-robust training techniques (data augmentation, regularization) can restore Cognitive Graph's performance advantage on noisy data.

**Key Finding**: Only heavy noise augmentation (50%) restores CG advantage at 1% noise level, achieving 6.94% improvement with 76% win rate. Lighter augmentation (10-20%) and regularization alone fail to make CG competitive.

**Implications**: 
1. CG's graph structure is fundamentally fragile to noise — GNN message passing amplifies noise while baseline concatenation treats features independently.
2. Making CG robust requires aggressive training (5x more noise in training than testing).
3. This explains why CG showed 81% improvement on clean synthetic data (H1.461) but failed on real robot data (H1.462).

**Next Step**: Test architectural changes (skip connections, batch norm, different GNNs) for better noise robustness, or apply 50% noise augmentation to real robot data to see if CG can finally match baseline performance.

**Status**: H1 is now CONDITIONAL — CG can beat baseline but only with heavy noise augmentation, suggesting practical deployment would require extensive data augmentation.