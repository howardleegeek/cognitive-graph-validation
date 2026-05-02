# H3.28: Temporal Consistency in Point Cloud

## Hypothesis Statement

Joint point cloud representation maintains temporal consistency across action sequences (smooth transitions) compared to frame-by-frame separate encoding.

## Parent Hypotheses

- H3.27: Joint Point Cloud (+97.9%) - JUST COMPLETED

## Research Context

H3.27 showed joint robot+scene representation achieves +97.9% improvement on cross-embodiment transfer. But that was STATIC - measured at a single timestep.

Real robot tasks require TEMPORAL consistency - actions are sequences of movements, not isolated poses.

## Experimental Design

| Configuration | Description |
|----------------|-------------|
| Separate (frame-by-frame) | Each timestep encoded independently |
| Joint + Temporal | Joint point cloud with temporal attention |
| Joint + Motion Model | Predict next point cloud from current |

### Evaluation

- Temporal smoothness: ||p_t+1 - p_t|| should be small
- Prediction accuracy: can predict next point cloud from history
- Cross-embodiment + temporal combined

## Expected Outcome

Joint + Temporal should outperform joint alone because:
1. Temporal attention captures motion patterns
2. Smoother point trajectories
3. Better dynamics modeling

## Status

**NEW** - Created based on H3.27 findings

## Prediction

**+10-30% improvement** over static joint representation.