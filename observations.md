## Key observations

**Context.** The base model already solves 62.2% of GSM8K (full test set), and no fine-tuned model in this project beat it (best: 0.609, the attention-only adapter at lr 2e-4). There was therefore no specialization gain to trade against forgetting. The experiments measure how much each setup damages the model and how well that damage can be repaired. Likely reasons, not tested: GSM8K's training solutions are terse, and the base model already reasons well on this task.

**Adapter placement (lr 2e-4, full test sets).** No adapter configuration moves MMLU, ARC, HellaSwag or Winogrande by more than about one point. GSM8K falls by about 5 points with MLP-only and both-layer adapters (0.622 to 0.575 and 0.573) but not with attention-only adapters (0.609, within noise), which train about a quarter of the parameters.

**Which capabilities are most fragile (stress setting, lr 1e-3, run A).**
- Most fragile: MMLU STEM (-11.1 points, -21% relative), MMLU social sciences (-10.3, -15%), ARC (-4.9, -11%) and MMLU other (-5.8, -9%). MMLU humanities (-3.7) is borderline. HellaSwag (0.0) and SciQ (-0.2) did not change, and Winogrande (-1.8) is within noise.
- They do not degrade at the same rate. MMLU fell abruptly: 0.617 at the base, 0.602 at step 25 (within noise), 0.523 at step 50, then it stayed between 0.51 and 0.54 for the rest of the 234 steps. ARC declined gradually (0.449, 0.424, 0.408, 0.400 at the final step). Winogrande moved late (0.648 at step 100, 0.623 at step 200).
- Hypothesis (not tested): quantitative and scientific recall (STEM, ARC, quantitative social-science subjects) overlaps more with writing arithmetic solutions than commonsense continuation (HellaSwag) does, so math training overwrites it first.
- Most of the MMLU damage appears by step 50, so very early stopping would avoid it. We did not evaluate GSM8K at those checkpoints, so the cost of stopping early is unknown.

**Data strategies (same GSM8K examples plus 3,200 general examples).** All three mixing variants recovered about 60 to 65% of the MMLU loss (0.537 without mixing to 0.584 to 0.589; base 0.617) with no cost to GSM8K (0.557 to 0.567 against 0.563). Shuffled and interleaved batches behave the same (MMLU 0.588 against 0.584). Matching the data to the fragile area mattered for ARC: with SciQ added, ARC rose to 0.488 (+3.9 over the base), while with Alpaca it stayed down (0.382 and 0.380), a gap of about 10 points, well beyond noise. STEM recovery was best with SciQ (0.515 against 0.502 and 0.487), but that difference is within noise. Side effect: Alpaca-based mixes lowered SciQ by about 3 points (0.908 and 0.910 against 0.940). No variant removed the MMLU drop entirely (about 3 points below base remain).

**Sequential specialization.** Stage 2 (adding science with about 29% math replay) raised ARC (0.400 to 0.488) and STEM (0.420 to 0.469) but cost math (GSM8K 0.563 to 0.473) and humanities (0.620 to 0.546). Stage 3 (adding general instruction data) restored MMLU to 0.594 (STEM 0.523, within noise of the base) but did not recover GSM8K (0.477). Across the stages GSM8K went 0.622, 0.563, 0.473, 0.477, so LoRA's small parameter footprint did not isolate the domains: every stage updates the same adapter. Training math and science jointly (run M3) kept GSM8K at 0.563 and reached ARC 0.488, so joint training retained more than sequential training, although that comparison is confounded by the math share of the final data (70% against 29%). Order was not tested.

**Adapter composition.** Three adapters were trained separately from the base model (math, science, general). Individually, the science adapter raised ARC (+4.8) and SciQ (+2.4) but wrecked math and knowledge (GSM8K 0.260, MMLU 0.341), and the general adapter was mild (MMLU -3.0, ARC -5.3). Merged in weight-change space:

| Merge | GSM8K | MMLU | ARC | SciQ |
|---|---:|---:|---:|---:|
| Base model | 0.622 | 0.617 | 0.449 | 0.942 |
| Arithmetic mean | 0.587 | 0.615 | 0.515 | 0.972 |
| Geometric mean (sign-agreement) | 0.597 | 0.626 | 0.487 | 0.963 |
| Harmonic mean (sign-agreement) | 0.607 | 0.625 | 0.491 | 0.962 |
| Sum (scale 1) | 0.310 | 0.354 | 0.520 | 0.964 |

All three mean-based merges kept the science gain (SciQ +2.0 to +3.0 over the base, ARC +3.8 to +6.6, at least as high as the science adapter alone for the arithmetic mean) while leaving math and knowledge at base level (GSM8K within the margin of the base, MMLU within about one point). The three means cannot be ranked: their differences are within noise. All of them beat the sequential model on math (0.587 to 0.607 against 0.477) and matched or exceeded it on MMLU and ARC. The geometric and harmonic variants keep only update entries on which all three adapters agree in sign, which gives a sparse update, yet they preserved the science gain; our guess (not tested) is that the sign-consistent entries carry the robust, shared directions. The sum with scale 1 collapsed math and knowledge (GSM8K 0.310, MMLU 0.354) while keeping the science gain: the science adapter's damage dominates when its update is added at full strength. "Retaining all specializations" is limited here because the math adapter had no gain to retain. SciQ for all merges and GSM8K for the geometric and harmonic merges were evaluated after a session restart, from adapter backups stored in half precision; the effect of that rounding is expected to be negligible.

## What worked, what did not, and what remains open

**Worked**
- Mixing general data into the math training, with any schedule, repaired about 60 to 65% of the MMLU loss at no GSM8K cost.
- Data matched to the fragile area (SciQ for the science and STEM weakness) was the only variant that protected ARC.
- Averaging separately trained adapters (arithmetic, geometric or harmonic mean) kept the science gain and kept math and general knowledge at base level, and beat sequential training on math.
- A final general-data stage in the sequential run restored MMLU.
- Attention-only adapters matched the other placements at lr 2e-4 with a quarter of the parameters and without the GSM8K loss.

**Did not work**
- No setting improved GSM8K over the base model.
- Interleaving batches gave no advantage over shuffling.
- Sequential training cascaded the loss in math despite replaying math examples.
- Task arithmetic with scale 1 collapsed math and knowledge.
- Alpaca mixing did not protect ARC and lowered SciQ slightly.

**Unresolved**
- Whether any math gain is possible with more informative data (for example MetaMathQA) or a stronger objective, and how merging behaves when an adapter does have a gain to retain.
- Order effects in sequential training, larger replay shares, and lower learning rates in later stages.
- A scale sweep for task arithmetic between 1/3 and 1.
- Why the science-only adapter destroyed math and knowledge (answer-format shift or genuine overwriting).
- The mechanism behind the abrupt MMLU drop before step 50, and what early stopping would cost on GSM8K.
- Single seed throughout, limited evaluation sets for the stress experiments, and one model (no other sizes or families).
