# H3.27: Joint Point Cloud Representation

## Hypothesis Statement

Joint point cloud (robot + scene) representation generalizes better than separate representations for cross-embodiment transfer.

## Parent Hypotheses

- H3.25: PointFlow (+92.2% cross-embodiment) - SUPPORTED
- H3.26: Action-conditioned point flow - REFUTED (-2.5%)

## Research Context

PointFlow achieved +92.2% on cross-embodiment transfer by representing state and action in unified 3D spatial domain. However, this was for "point displacement" - it tracks how points move.

Joint point cloud goes further: represent BOTH robot end-effector AND scene as a SINGLE unified point cloud, enabling embodiment-agnostic dynamics learning.

## Literature Reference

- **Unified World Models** (arXiv:2504.02792): Coupling Video and Action Diffusion
- **PointWorld** (ICLR 2026): Unifying state and action in shared 3D spatial domain

## Experimental Design

### Point Cloud Configurations

| Configuration | Robot Points | Scene Points | Fusion |
|--------------|-------------|--------------|--------|
| Separate (baseline) | 256 (robot only) | 1024 (scene only) | Concatenation |
| Joint | 512 (robot + scene combined) | 512 (scene) | Unified |
| Unified | 1024 (joint) | N/A | Single representation |

### Evaluation

- Cross-embodiment transfer MSE
- Zero-shot generalization to unseen robot platforms (different DOF, reach, payload)

## Expected Outcome

Joint point cloud should outperform separate because:
1. Learns embodiment-agnostic dynamics
2. No need to map between different robot configurations
3. Shared representation space

## Status

**PENDING** - Ready for execution

## Prediction

**Joint wins by +5-15%** over separate representations.