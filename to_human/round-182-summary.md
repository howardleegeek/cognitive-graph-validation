# Round 182 Summary — H1.414: Temporal CG with Recurrent Message Passing

## Experiment

Designed and ran H1.414 to test whether explicit temporal modeling (recurrent message passing) would maintain CG advantage over longer planning horizons, addressing the H1.413 finding that CG advantage decreases with sequence length. Built a Temporal-CG architecture that processes actions one timestep at a time with GRU-style recurrent updates, compared against baseline (flat MLP) and original CG (flat input) on a 5-object physics prediction task with up to 10 action steps.

## Results

**REFUTED.** Temporal-CG performed dramatically worse than both baselines across all sequence lengths (e.g., -788% improvement at 1 step, -157% at 10 steps). However, a striking inverse scaling pattern emerged: Temp-CG loss *decreases* with sequence length (0.258 at 1 step → 0.014 at 10 steps), the opposite of what both baseline and CG show. This suggests the recurrent architecture is significantly harder to train under current conditions (40 epochs) rather than being fundamentally flawed. The self-recurrent GRU formulation (h_new = GRU(h, h)) may not provide meaningful temporal dynamics.

## Next Steps

Three follow-up experiments planned: (1) H1.415 — train Temp-CG for 200+ epochs to check convergence, (2) H1.416 — try proper GRU with separate input/hidden states, (3) H1.417 — curriculum learning approach (short→long sequences). The inverse loss scaling pattern is intriguing enough to warrant deeper investigation.
