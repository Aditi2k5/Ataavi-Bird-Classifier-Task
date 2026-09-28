# Bird Call Classifier — Write-up

**Task & data.** Six Indian species from real field recordings. Five sourced from
Xeno-canto (IDs in `data/recordings_used.csv`); Oriental Magpie-Robin was
download-restricted on Xeno-canto during the project and supplied directly by the
team. Grades B–E were kept on purpose to include faint, noisy, and overlapping
audio. Imbalance is intentional: 70 recordings each for the common species, 22 for
magpie-robin, and **7 for the eagle-owl** — the rare class the task centres on.

**Approach & why (step 2).** Frozen **BirdNET** embeddings (1024-d per 3 s window)
+ a class-balanced **logistic-regression** head. A from-scratch CNN would overfit a
7-recording class; BirdNET, trained on Xeno-canto-scale field audio, gives a
noise-robust representation in which even a rare class is separable from few
examples. *BirdNET already handles* the hard perception (noisy audio → a
species-relevant, overlap-robust representation, all six species in-vocabulary).
*We added* the calibrated head, the imbalance handling, the multi-label and
precision analyses, and honest evaluation under scarcity. We compared SVM/kNN/MLP
heads — none beat balanced LR — so we kept the simplest interpretable head
(avoiding needless capacity is itself anti-overfitting).

**Handling the messy parts (step 3).** *Imbalance:* balanced class weights +
recording-level `StratifiedGroupKFold` + owl context-augmentation; a leakage audit
verifies no recording crosses folds (all checks pass, 309 recordings). *Faint:*
loudness normalisation; reported as its own slice. *Overlap:* real multi-label
detection on synthetic 2-species mixtures (one-vs-rest head). *Scarcity (owl):* a
precision-first decision threshold.

**Results (step 4).** 10-seed × 5-fold recording-level CV: **macro-F1 0.840 ± 0.027,
accuracy 0.897 ± 0.006**. Per-class F1: cisticola 0.97, magpie 0.91, cuckoo 0.89,
sunbird 0.89, bulbul 0.87, **eagle-owl 0.52 ± 0.15**. Confusion matrix:
`outputs/final_confusion_matrix.png`. Where it struggles, with one unifying cause —
**spectral complexity**: (a) the owl (recall 0.55) is confused mainly with the
acoustically similar cuckoo (shown irreducible ~30%); a precision-first threshold
gives **93% precision at 53% recall**. (b) Faint audio drops macro-F1 0.88→0.63,
hitting broadband/melodic species hard, cisticola nearly immune. (c) Overlap: the
foreground bird is caught ~0.78, both birds ~0.55 at equal loudness, falling to 0.38
as the second bird quietens — the *same* spectral law as faintness, with the owl the
first to vanish. All variance localises to the owl (n=7); the method itself is
stable (±0.006 accuracy).

**Honesty notes.** We report the CV interval, not the best single split (0.859).
Owl rests on 7 recordings (wide CI, reported not smoothed). Magpie-robin is
team-provided, segmented, and lacks overlap metadata. Overlap is evaluated on
synthetic mixtures because real clips lacked per-species labels.

**With more time.** (1) Perch embeddings — the owl-cuckoo overlap is a property of
*BirdNET's* space; a different model might separate them. (2) A dedicated
multi-label head with per-class thresholds/mAP. (3) Faint-specific mitigation
(normalise before embedding; attenuation augmentation). (4) More owl data / few-shot
methods. (5) An abstain option for the confident owl-cuckoo ambiguous windows.

*(A detailed version with all supporting experiments is in `ANALYSIS.md`.)*