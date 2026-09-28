# Bird Call Classifier — 6 Indian species

Classifies short audio clips into six species using real field recordings, with
deliberate handling of faint audio, background noise, overlapping calls, and a
large class imbalance. **BirdNET frozen embeddings + a class-balanced
logistic-regression head.**

Species: Indian Cuckoo, Indian Eagle-Owl, Oriental Magpie-Robin, Purple-rumped
Sunbird, Zitting Cisticola, Red-vented Bulbul.

## Headline result
10-seed × 5-fold **recording-level** CV: **macro-F1 0.840 ± 0.027, accuracy
0.897 ± 0.006**. Five classes stable to ±0.01; the eagle-owl (0.52 ± 0.15, n=7
recordings) carries essentially all variance — scarcity, localised and quantified.
Full analysis in `WRITEUP.md`; figure in `outputs/final_confusion_matrix.png`.

Pipeline is leakage-audited (`src/audit_leakage.py`, all checks pass on 309
recordings) and stability-checked (`src/exp_repeated_cv.py`).


## Submission contents
- `WRITEUP.md` — half-page write-up (approach + reasoning for steps 2–4 + next steps)
- `ANALYSIS.md` — full analysis with all supporting experiments (appendix)
- `data/recordings_used.csv` — recordings used (XC IDs + tags; `source` column marks
  the team-provided magpie-robin, which XC had download-restricted)
- `outputs/final_confusion_matrix.png` + `final_metrics.json` — headline metrics
- `outputs/multilabel_summary.txt` — overlap multi-label results
- `src/` — full pipeline + experiment scripts, `requirements.txt`, `config.yaml`

## Data notes
- Five species from Xeno-canto (IDs + tags in `data/recordings_used.csv`), grades
  B–E kept on purpose to include faint/noisy/overlapping audio.
- Oriental Magpie-Robin was **download-restricted on Xeno-canto**; recordings were
  team-provided (segmented; no overlap metadata, so excluded from the overlap slice).
- Imbalance is intentional: 70 recordings/common species, 22 magpie, **7 owl**.

## Setup
```bash
pip install -r requirements.txt
# for the 5 Xeno-canto species (magpie-robin is supplied separately):
export XENO_CANTO_API_KEY="your-key"      # or put it in .env
```
BirdNET embedding extraction needs TensorFlow (`BirdNETEmbedder` reads the
intermediate embedding tensor directly, so it works with plain `tensorflow` — no
`tflite_runtime`). Audio decoding of some MP3s needs `ffmpeg` (`brew install ffmpeg`).

## Run
```bash
python src/download_data.py            # fetch 5 XC species -> data/audio + manifest
# (place team-provided magpie-robin WAVs in data/audio/oriental_magpie_robin/)
python src/extract_all.py --list-only  # sanity-check grouping/dedup, no TF
python src/extract_all.py              # BirdNET -> outputs/birdnet_embeddings.npz
python src/baseline_birdnet.py --from-embeddings outputs/birdnet_embeddings.npz
python src/audit_leakage.py            # verify no leakage
python src/exp_repeated_cv.py          # confidence intervals
python src/analyze.py                  # per-slice, per-class breakdown
```

## Key scripts
```
src/download_data.py     Xeno-canto v3 download (content-validated, over-fetch)
src/extract_all.py       format-agnostic BirdNET extraction, segment grouping, dedup
src/baseline_birdnet.py  BirdNETEmbedder (direct tensor read) + balanced-LR CV eval
src/audit_leakage.py     asserts recording-level split has no leakage
src/exp_repeated_cv.py   10-seed CV -> mean ± std per class
src/analyze.py           clean/faint/overlap slices, per-class
src/exp_heads.py         head comparison (LR vs SVM/kNN/MLP) — justifies keeping LR
src/exp_prototype.py     few-shot nearest-centroid experiment (owl)
src/exp_twostage.py      owl-vs-cuckoo two-stage rescuer
src/exp_owl_threshold.py owl decision-threshold sweep
src/splits.py, metrics.py  shared recording-level split + metrics/plots
```

## What we found (short version)
- Strong, stable on common species; magpie-robin easy (0.91) but from a clean,
  small team set.
- **Owl** is the bottleneck: recall 0.55, and 21/34 errors go to the acoustically
  similar cuckoo — shown five ways to decompose into a recoverable (imbalance) and
  an irreducible (~30% embedding overlap) part.
- **Faint** audio is a broad tax proportional to spectral complexity (cisticola
  immune, broadband melodic callers hit hard).
- **Overlap** is essentially free — BirdNET embeddings are overlap-robust.