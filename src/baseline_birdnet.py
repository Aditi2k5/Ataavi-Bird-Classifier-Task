from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, set_seed
from metrics import compute_and_save_metrics

def _robust_load(path, sr):
    """Load audio at `sr` mono. Falls back to pydub for files with junk/leading
    bytes that libsndfile/audioread can't skip (some Xeno-canto MP3s)."""
    import numpy as np
    try:
        import librosa
        return librosa.load(str(path), sr=sr, mono=True)[0]
    except Exception:
        pass
    from pydub import AudioSegment          # pip install pydub
    seg = AudioSegment.from_file(str(path)).set_channels(1).set_frame_rate(sr)
    y = np.array(seg.get_array_of_samples()).astype(np.float32)
    return y / (float(1 << (8 * seg.sample_width - 1)) + 1e-9)   # int -> [-1,1]

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def _locate_birdnet_model() -> str:
    """Find the BirdNET .tflite model. Uses birdnetlib's Analyzer (which loads
    fine even where its embedding method is broken) purely to get the path."""
    from birdnetlib.analyzer import Analyzer
    a = Analyzer()
    for attr in ("model_path", "MODEL_PATH", "model"):
        p = getattr(a, attr, None)
        if isinstance(p, str) and p.endswith((".tflite", ".fp32")):
            return p
    # fallback: search the installed birdnetlib package
    import birdnetlib, glob
    root = str(Path(birdnetlib.__file__).parent)
    hits = glob.glob(root + "/**/*.tflite", recursive=True)
    hits = [h for h in hits if "Meta" not in h] or hits
    if hits:
        return hits[0]
    raise SystemExit("Could not locate the BirdNET .tflite model via birdnetlib.")


class BirdNETEmbedder:
    """Runs the BirdNET TFLite model directly and reads the 1024-d embedding
    tensor. The embedding is an INTERMEDIATE tensor (not a model output), which
    current TFLite frees during invoke() and the XNNPACK delegate fuses away —
    that's the 'Tensor data is null' error birdnetlib and naive code both hit.
    We fix it by (a) preserving all tensors and (b) disabling the default
    delegate, then locating the 1024-d tensor and validating on a dummy input."""

    def __init__(self, sample_rate: int = 48000):
        import tensorflow as tf
        model_path = _locate_birdnet_model()
        base = dict(model_path=model_path, num_threads=4,
                    experimental_preserve_all_tensors=True)
        try:   # disable XNNPACK so intermediate tensors aren't fused away
            self._it = tf.lite.Interpreter(
                **base,
                experimental_op_resolver_type=(
                    tf.lite.experimental.OpResolverType
                    .BUILTIN_WITHOUT_DEFAULT_DELEGATES))
        except Exception:
            self._it = tf.lite.Interpreter(**base)
        self._it.allocate_tensors()

        ind = self._it.get_input_details()[0]
        self.in_idx = ind["index"]
        self.sr = sample_rate
        last = int(ind["shape"][-1])
        self.chunk = last if last > 1 else sample_rate * 3
        out_idx = self._it.get_output_details()[0]["index"]

        # dummy invoke, then find the 1024-d embedding tensor by reading it back
        self._it.set_tensor(self.in_idx,
                            np.zeros((1, self.chunk), dtype=np.float32))
        self._it.invoke()
        cands = []
        for d in self._it.get_tensor_details():
            i = d["index"]
            try:
                t = self._it.get_tensor(i)
            except Exception:
                continue
            if t.ndim == 2 and t.shape[-1] == 1024:
                cands.append(i)
        if not cands:
            shapes = [(d["index"], list(d["shape"])) for d in
                      self._it.get_tensor_details()]
            raise SystemExit("No readable 1024-d embedding tensor found. Tensor "
                             f"shapes: {shapes}")
        below = [c for c in cands if c < out_idx]
        self.emb_idx = max(below) if below else max(cands)
        print(f"[BirdNETEmbedder] input idx={self.in_idx} chunk={self.chunk} "
              f"emb_idx={self.emb_idx} (validated 1024-d)")

    def embed_signal(self, sig: np.ndarray):
        """Yield one embedding per non-overlapping 3 s chunk (last chunk padded)."""
        import math
        out = []
        n = max(1, math.ceil(len(sig) / self.chunk))
        for i in range(n):
            seg = sig[i * self.chunk:(i + 1) * self.chunk]
            raw_len = len(seg)
            if raw_len < self.chunk:
                seg = np.pad(seg, (0, self.chunk - raw_len))
            self._it.set_tensor(self.in_idx, seg.reshape(1, -1).astype(np.float32))
            self._it.invoke()
            emb = np.asarray(self._it.get_tensor(self.emb_idx)[0], dtype=np.float32)
            t0 = i * self.chunk / self.sr
            out.append({"embeddings": emb, "start_time": t0,
                        "end_time": t0 + self.chunk / self.sr,
                        "rms": _rms(sig[i * self.chunk:i * self.chunk + raw_len])})
        return out


# --------------------------------------------------------------------------- #
# 1. embedding extraction (needs birdnetlib + audio; run where TF works)
# --------------------------------------------------------------------------- #
def extract_embeddings(cfg) -> dict:
    import pandas as pd
    import librosa

    bn = cfg["birdnet"]
    sr = bn["sample_rate"]
    embedder = BirdNETEmbedder(sample_rate=sr)
    man = pd.read_csv(cfg["paths"]["manifest"])
    classes = [s["label"] for s in cfg["species"]]

    X, y, groups, rms_vals, overlap = [], [], [], [], []
    n_ok, n_fail = 0, 0
    for _, r in man.iterrows():
        label, xc = r["label"], str(r["xc_id"])
        path = Path(cfg["paths"]["audio_dir"]) / label / f"{xc}.mp3"
        if not path.exists():
            print(f"  ! missing {path}"); n_fail += 1; continue
        try:
            sig = _robust_load(path, sr)
            windows = embedder.embed_signal(sig)
        except Exception as e:
            print(f"  ! embed failed {label} XC{xc}: {e}"); n_fail += 1; continue
        for w in windows:
            X.append(w["embeddings"])
            y.append(classes.index(label)); groups.append(xc)
            overlap.append(bool(r.get("has_overlap", False)))
            rms_vals.append(w["rms"])
        n_ok += 1
        print(f"  {label:24s} XC{xc}: {len(windows)} windows")

    if not X:
        raise SystemExit(f"No embeddings extracted ({n_fail} recordings failed).")
    print(f"\nextracted {n_ok} recordings ok, {n_fail} failed")

    rms_vals = np.asarray(rms_vals, float)
    thr = np.nanpercentile(rms_vals, bn["faint_rms_percentile"])
    out = {"X": np.stack(X), "y": np.asarray(y), "groups": np.asarray(groups),
           "is_faint": (rms_vals <= thr), "has_overlap": np.asarray(overlap),
           "classes": np.asarray(classes)}
    dest = Path(cfg["paths"]["out_dir"]) / "birdnet_embeddings.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    print(f"[cached] {dest}  X={out['X'].shape}, faint={int(out['is_faint'].sum())}, "
          f"overlap={int(out['has_overlap'].sum())}")
    return out


# --------------------------------------------------------------------------- #
# 2. context augmentation for the owl (real owl call (+) real background)
#    Audio-domain, embedded via RecordingBuffer. Each sample is tagged with its
#    source owl recording AND its background recording, so the k-fold loop can
#    include an augmented sample only when BOTH sources are in that fold's TRAIN
#    set -> no test-acoustics leakage.
# --------------------------------------------------------------------------- #
def context_augment_owl(cfg) -> dict:
    import pandas as pd
    import librosa

    bn, ag = cfg["birdnet"], cfg["augment"]
    sr = bn["sample_rate"]
    seg = int(ag["seg_seconds"] * sr)
    rng = np.random.default_rng(cfg["seed"])
    embedder = BirdNETEmbedder(sample_rate=sr)

    man = pd.read_csv(cfg["paths"]["manifest"])
    owl_label = bn["owl_label"]
    owl_rows = man[man.label == owl_label]
    bg_rows = man[man.label != owl_label]           # backgrounds = other species

    def load(label, xc):
        p = Path(cfg["paths"]["audio_dir"]) / label / f"{xc}.mp3"
        try:
            s, _ = librosa.load(str(p), sr=sr, mono=True); return s
        except Exception:
            return None

    def rand_seg(sig):
        if sig is None or len(sig) < seg:
            return None
        st = rng.integers(0, len(sig) - seg + 1)
        return sig[st:st + seg]

    X_aug, owl_src, bg_src = [], [], []
    for _, orow in owl_rows.iterrows():
        owl_sig = load(owl_label, str(orow.xc_id))
        for _ in range(ag["n_aug_per_owl"]):
            oseg = rand_seg(owl_sig)
            if oseg is None:
                continue
            brow = bg_rows.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            bseg = rand_seg(load(brow.label, str(brow.xc_id)))
            if bseg is None:
                continue
            snr = rng.uniform(*ag["snr_db"])
            # scale background to the target SNR relative to the owl segment
            po, pb = _rms(oseg) ** 2 + 1e-12, _rms(bseg) ** 2 + 1e-12
            g = np.sqrt(po / (pb * (10 ** (snr / 10))))
            mix = oseg + g * bseg
            mix = mix / (np.max(np.abs(mix)) + 1e-8)
            for w in embedder.embed_signal(mix):
                X_aug.append(w["embeddings"])
                owl_src.append(str(orow.xc_id)); bg_src.append(str(brow.xc_id))

    out = {"X_aug": np.stack(X_aug) if X_aug else np.zeros((0, 1024), np.float32),
           "owl_src": np.asarray(owl_src), "bg_src": np.asarray(bg_src)}
    dest = Path(cfg["paths"]["out_dir"]) / "birdnet_owl_aug.npz"
    np.savez_compressed(dest, **out)
    print(f"[cached] {dest}  X_aug={out['X_aug'].shape}")
    return out


# --------------------------------------------------------------------------- #
# 3. k-fold fit + honest evaluation (numpy/sklearn only -> testable anywhere)
# --------------------------------------------------------------------------- #
def kfold_fit_eval(data, out_dir, n_folds=5, seed=1337, aug=None,
                   owl_index=None, min_conf=0.0, prefix="birdnet_",
                   title_suffix="BirdNET+balanced-LR (CV out-of-fold)"):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import f1_score

    X = np.asarray(data["X"], np.float32)
    y = np.asarray(data["y"]); groups = np.asarray([str(g) for g in data["groups"]])
    classes = [str(c) for c in data["classes"]]
    n = len(y)

    oof_pred = np.full(n, -1)
    per_fold_macro = []
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for k, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        Xtr, ytr = X[tr], y[tr]
        n_aug = 0
        # add augmented owl samples whose owl_src AND bg_src are both in train
        if aug is not None and aug["X_aug"].shape[0] and owl_index is not None:
            train_groups = set(groups[tr])
            keep = np.array([os in train_groups and bs in train_groups
                             for os, bs in zip(aug["owl_src"], aug["bg_src"])])
            if keep.any():
                Xtr = np.vstack([Xtr, aug["X_aug"][keep]])
                ytr = np.concatenate([ytr, np.full(keep.sum(), owl_index)])
                n_aug = int(keep.sum())
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0))
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(X[te])
        # map proba columns (clf.classes_) back to global class indices
        pred = clf.classes_[proba.argmax(1)]
        oof_pred[te] = pred
        fold_f1 = f1_score(y[te], pred, average="macro", zero_division=0)
        per_fold_macro.append(fold_f1)
        aug_note = f" (+{n_aug} owl aug)" if n_aug else ""
        print(f"  fold {k}: n_test={len(te)}  macroF1={fold_f1:.3f}{aug_note}")

    slice_masks = {}
    if "is_faint" in data:   slice_masks["faint"] = np.asarray(data["is_faint"])
    if "has_overlap" in data: slice_masks["overlap"] = np.asarray(data["has_overlap"])

    metrics = compute_and_save_metrics(
        y, oof_pred, classes, out_dir=out_dir, prefix=prefix,
        slice_masks=slice_masks, title_suffix=title_suffix)
    metrics["cv_macro_f1_mean"] = float(np.mean(per_fold_macro))
    metrics["cv_macro_f1_std"] = float(np.std(per_fold_macro))
    print(f"\nCV macro-F1 across folds = {metrics['cv_macro_f1_mean']:.3f} "
          f"± {metrics['cv_macro_f1_std']:.3f}  (spread reflects owl scarcity)")
    return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--from-embeddings", default=None,
                    help="skip extraction; load a cached .npz")
    ap.add_argument("--augment-owl", action="store_true",
                    help="include context-augmented owl samples (ablation toggle)")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = cfg["paths"]["out_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if args.from_embeddings:
        d = np.load(args.from_embeddings, allow_pickle=True)
        data = {k: d[k] for k in d.files}
    else:
        data = extract_embeddings(cfg)
    if args.extract_only:
        return

    aug = None
    if args.augment_owl:
        ap_path = Path(out_dir) / "birdnet_owl_aug.npz"
        if ap_path.exists():
            d = np.load(ap_path, allow_pickle=True); aug = {k: d[k] for k in d.files}
        else:
            aug = context_augment_owl(cfg)

    classes = [str(c) for c in data["classes"]]
    owl_index = classes.index(cfg["birdnet"]["owl_label"])
    prefix = "birdnet_aug_" if args.augment_owl else "birdnet_"
    kfold_fit_eval(data, out_dir, n_folds=cfg["eval"]["n_folds"], seed=cfg["seed"],
                   aug=aug, owl_index=owl_index, min_conf=args.min_conf, prefix=prefix)


if __name__ == "__main__":
    main()