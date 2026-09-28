# Bird Call Classifier — Write-up

## Task & dataset
Six Indian species classified from short field recordings. Five sourced from
Xeno-canto (Indian Cuckoo, Indian Eagle-Owl, Purple-rumped Sunbird, Zitting
Cisticola, Red-vented Bulbul); Oriental Magpie-Robin was **download-restricted on
Xeno-canto** during the project and was supplied directly by the team. We
deliberately did **not** filter to quality A — grades B–E were kept to include
faint/noisy audio, and recordings with background species (Xeno-canto `also`
field) were kept as overlap cases. The class imbalance is intentional and real:
**70 recordings** each for the four common species, **22** for the magpie-robin
(team set, segmented into 52 clips), and only **7** for the eagle-owl — the rare
class the task is built around. Recording IDs are in `data/recordings_used.csv`.

## Approach (step 2) and why
**Frozen BirdNET embeddings + a class-balanced logistic-regression head.** Each
3 s window → BirdNET's 1024-d embedding → balanced LR over the 6 classes.

*Why embeddings, not a from-scratch CNN:* with 7 owl recordings, a CNN would
overfit the rare class hopelessly. BirdNET was trained on Xeno-canto-scale field
audio, so its embeddings already encode noise-robust, species-relevant structure —
the transfer-learning regime that lets a rare class be modelled from a handful of
examples.

*What BirdNET already handles:* a robust representation of field audio (noise,
some overlap), and it already contains all six species. *What we added:* the head
calibrated to these labels, the imbalance handling, honest evaluation under
scarcity, and the analysis decomposing where and why it fails.

*Why the head stayed simple (anti-overfitting):* we compared LR against SVM-RBF,
kNN, and an MLP under identical CV — **none beat balanced LR** on either macro-F1
or owl-F1 (`src/exp_heads.py`). The ceiling is the data, not the classifier, so we
kept the simplest interpretable head rather than adding capacity that buys nothing.

## Handling the messy parts (step 3)
- **Class imbalance.** Recording-level `StratifiedGroupKFold` (never window-level),
  balanced class weights, and transfer learning so the rare class needs little
  data. A leakage audit (`src/audit_leakage.py`) verifies no recording crosses
  folds and each is tested exactly once — **all checks pass** on 309 recordings.
- **Scarcity (eagle-owl), attacked several ways.** Balanced weights, context
  augmentation (real owl call + real background), prototype/nearest-centroid, a
  two-stage owl-vs-cuckoo rescuer, and per-class thresholds. Findings below.
- **Faint / low-volume.** Per-window loudness normalisation; faint windows (bottom
  RMS quartile) are reported as a **separate slice** rather than hidden.
- **Overlapping calls.** Kept via the `also` field and reported as a **separate
  slice**; magpie-robin has no such metadata so it is correctly excluded from the
  overlap slice, not guessed.

## Results (step 4)
**10-seed × 5-fold recording-level CV (honest headline):**
- **macro-F1 = 0.840 ± 0.027,  accuracy = 0.897 ± 0.006.**
- Per-class F1 (mean ± std): cisticola 0.972 ± 0.004, magpie 0.910 ± 0.008,
  cuckoo 0.889 ± 0.005, sunbird 0.887 ± 0.008, bulbul 0.867 ± 0.011,
  **eagle-owl 0.516 ± 0.153**.

**The single most important result:** every class is stable to ±0.01 *except the
owl* (±0.153), which alone drives essentially all system variance. The method is
stable; the uncertainty is localised precisely to the class with 7 recordings —
scarcity, quantified. Confusion matrix: `outputs/final_confusion_matrix.png`.

**Where it struggles, and why:**
- *Owl (the rare class).* Recall 0.55; **21 of 34 owl errors go to the cuckoo.**
  Both are low-frequency, tonal, repetitive callers, so they sit close in
  BirdNET's embedding. We showed this five independent ways: multiclass confusion,
  an owl-vs-cuckoo binary (ceiling 0.77), a threshold sweep (recall flatlines —
  the misses are *confident*, not marginal), prototypes (0.92 recall but precision
  collapses), and a two-stage rescuer (cannot recover the confident-cuckoo owls).
  The owl difficulty decomposes into a **recoverable part** (imbalance at the
  owl–cuckoo boundary) and an **irreducible part** (~30% acoustically entangled in
  the embedding). Context augmentation shifted recall 0.55→0.68 but traded
  precision, leaving F1 ~flat — it rebalances rather than adds information.
- *Faint audio is a broad tax, not owl-specific.* Clean→faint per class: cisticola
  0.98→0.94 (nearly immune), cuckoo 0.92→0.79, sunbird/bulbul ~0.93→0.66–0.69,
  owl 0.59→0.30, magpie 0.95→0.40. The drop tracks **spectral complexity**:
  narrow-band, repetitive calls (cisticola) survive faintness; broadband melodic
  calls (bulbul, magpie, sunbird) wash out. A real, mechanistic finding.
- *Overlap is essentially free.* Overlap-slice macro-F1 (0.887) ≈ clean (0.883),
  per class. BirdNET's embedding is robust to background species — the foreground
  dominates the representation.


## Precision-first operating point for the rare owl
Because a false owl detection is costlier than a miss for a rare species, we set a
confidence threshold on the owl class. Sweeping it (`src/exp_owl_precision.py`):

| owl threshold | precision | recall | F1 | false-owls |
|---|---|---|---|---|
| 0.50 (argmax) | 0.82 | 0.55 | 0.66 | 9 |
| 0.60 | 0.89 | 0.55 | 0.68 | 5 |
| **0.70** | **0.93** | **0.53** | **0.68** | **3** |
| 0.90 | 0.94 | 0.45 | 0.61 | 2 |

We adopt **threshold 0.70: 93% precision at 53% recall**. Raising the bar from
argmax to 0.70 removes low-confidence false positives at negligible recall cost;
pushing higher starts dropping real owls, because the remaining errors are the
irreducible owl-cuckoo confident-misses. So when the model reports an eagle-owl it
is right 93% of the time, catching just over half of all owls. (Threshold tuned on
CV data — a deployment would fix it on a held-out fold; n=7, so we report the
trade-off shape, not a guarantee.)


## Overlap handled as multi-label detection (synthetic mixtures)
Beyond the robustness slice above, we tested true overlap: mixing two held-out
recordings of different species (labels known by construction) and asking a
one-vs-rest head for the *set* of species present, across mixing SNRs
(`src/exp_multilabel.py`, tau=0.15):

| 2nd bird quieter by | 0 dB | 6 dB | 12 dB |
|---|---|---|---|
| both birds detected | 0.55 | 0.48 | 0.38 |
| primary (louder)    | 0.75 | 0.78 | 0.82 |
| secondary (quieter) | 0.79 | 0.69 | 0.55 |
| false extra species | 0.07 | 0.07 | 0.07 |

The foreground bird is caught reliably (~0.78) and both birds ~55% of the time at
equal loudness; detection of the *second* bird falls monotonically as it gets
quieter. Per-species secondary detection at 0 dB (cisticola 37/40, cuckoo 37/40,
sunbird 36, bulbul 32, magpie 30, **eagle-owl 17**) follows the **same
spectral-complexity ordering as the faint slice** — narrow-band repetitive calls
survive being the quiet one, broadband/rare ones wash out. The owl is detectable in
overlap when equally loud (17/40) but collapses fastest as it quietens
(→ 8/40 at 12 dB): overlap failure is, mechanistically, the faintness effect.
(Synthetic mixtures — real overlap may differ, but real clips lacked per-species
labels; owl/magpie mixture counts are small, so treat as indicative.)

## Honesty / caveats
- Owl F1 rests on **7 recordings**; its wide CI is reported, not smoothed over.
- Magpie-robin is team-provided, segmented, dominated by 2 of its 22 recordings,
  and lacks overlap metadata — its high score may generalise less than cisticola's.
- BirdNET was trained on Xeno-canto, so it may have seen recordings near ours; this
  affects all embedding-based bird work and is noted, not hidden.
- We report the **CV interval**, not the best single split (which reads 0.859).

## With more time
1. **Perch embeddings** — the owl–cuckoo floor is a property of *BirdNET's* space;
   a different foundation model might separate them. Highest-upside next test.
2. **Multi-label overlap head** with per-class thresholds and mAP, since overlap is
   inherently multi-label.
3. **Faint-specific mitigation** — loudness-normalise before embedding; train-time
   attenuation augmentation targeting the broadband classes.
4. **More owl data / few-shot methods** — the one lever that moves the rare class.
5. **Abstain / coverage** — let the model decline the confident owl–cuckoo
   ambiguous windows, turning the irreducible floor into a calibrated "unsure".