# Progress Report — Cognitive Graph Validation

**Cycle 80 — May 2, 2026**

## New Experiment Completed

### H3.25: PointFlow Representation (NEW - Literature-Based)

Based on ICLR 2026 paper **PointWorld** (arXiv:2601.03782) - unifying state and action in shared 3D spatial domain.

| Metric | PointFlow (3D Spatial) | Concatenation (512D) | Improvement |
|--------|----------------------|---------------------|-------------|
| Cross-Embodiment Transfer MSE | 0.0306 | 0.3900 | **+92.2%** |

**Result: ✅ SUPPORTED** — PointFlow representation dramatically outperforms on cross-embodiment transfer!

Key insight from PointWorld: Represent both state (scene points) and action in same 3D spatial domain as "point flow" enables zero-shot transfer across different robot embodiments (Franka → bimanual) without fine-tuning.

### Literature Referenced
- **PointWorld**: Scaling 3D World Models for In-The-Wild Robotic Manipulation (ICLR 2026)
- **Unified World Models**: Coupling Video and Action Diffusion (arXiv:2504.02792)
- **Ctrl-World**: Controllable Generative World Model for Robot Manipulation

## Research Status

| # | Hypothesis | Status | Key Finding |
|---|------------|--------|-------------|
| H1 | Unified vs Baseline | ✅ +25.6% | Early fusion wins |
| H1.102 | Unified + SSM | ✅ +28.9% | Combined best |
| H2.x | Graph structure | ✅ | +56-75% on temporal |
| H3.8 | SSM > Attention | ✅ +93% | Long sequences |
| H3.9 | Mamba > Attention | ✅ +92.8% | Gated mechanism |
| H3.25 | PointFlow representation | ✅ +92.2% | Cross-embodiment transfer |
| H3.24 | Attention 20+ seq | ⚠️ +5.7% | Wins at 30 only |

**Total: 26+ SUPPORTED, 1 INCONCLUSIVE, 13 REFUTED**

## Key Takeaways

1. **Unified architecture** (+25.6% real robot) validated
2. **SSM/Mamba** (+92.8%) outperforms attention on long sequences
3. **Graph structure** (+75%) excels at temporal reasoning
4. **Attention** marginal, concatenation preferred for most tasks
5. **NEW: PointFlow** (+92.2%) enables cross-embodiment transfer

## Next Directions

1. Test H3.26: Action-conditioned point flow
2. Paper draft: ICRA/RSS structure
3. Consolidate paper-ready findings

---
*Never stop. Always experimenting.*