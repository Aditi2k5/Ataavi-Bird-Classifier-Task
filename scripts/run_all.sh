#!/usr/bin/env bash
# End-to-end pipeline. Assumes XENO_CANTO_API_KEY is set. Edit approach in config.yaml.
set -euo pipefail
python src/download_data.py                 # -> data/audio, data/recordings_used.csv
# --- primary approach ---
python src/baseline_birdnet.py              # -> outputs/birdnet_*  [wip]
# --- OR the CNN approach ---
# python src/preprocess.py                  # -> data/cache        [wip]
# python src/train.py                       # -> outputs/cnn_model.pt [wip]
# python src/evaluate.py --approach cnn     # -> outputs/cnn_*      [wip]
