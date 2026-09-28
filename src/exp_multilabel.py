from __future__ import annotations
import sys, argparse
from pathlib import Path
from collections import defaultdict
from itertools import permutations
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, set_seed
from splits import recording_level_split

def energetic_segment(sig, seg):
    if len(sig) <= seg:
        return np.pad(sig, (0, seg - len(sig)))
    hop = seg // 2; best, best_rms = sig[:seg], -1.0
    for s in range(0, len(sig) - seg + 1, hop):
        w = sig[s:s+seg]; r = np.sqrt(np.mean(w**2))
        if r > best_rms: best_rms, best = r, w
    return best

def mix_at_snr(a, b, snr_db):
    pa = np.mean(a**2)+1e-12; pb = np.mean(b**2)+1e-12
    gain = np.sqrt(pa/(pb*(10**(snr_db/10))))
    m = a + gain*b
    return (m/(np.max(np.abs(m))+1e-8)).astype(np.float32)

def predict_set(row, names, tau):
    return {names[i] for i,p in enumerate(row) if p >= tau}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n-per-pair", type=int, default=8)
    ap.add_argument("--snrs", type=float, nargs="+", default=[0.0, 10.0])
    ap.add_argument("--taus", type=float, nargs="+", default=[0.3, 0.5])
    ap.add_argument("--secondary-any", action="store_true",
                    help="draw the quieter bird from ALL recordings, not just test (more owl/magpie coverage)")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"])
    import pandas as pd, librosa
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from baseline_birdnet import BirdNETEmbedder

    d = np.load(Path(cfg["paths"]["out_dir"])/"birdnet_embeddings.npz", allow_pickle=True)
    X = d["X"].astype("float32"); y = d["y"]; g = np.array([str(x) for x in d["groups"]])
    classes = [str(c) for c in d["classes"]]; labels = np.array(classes)[y]

    splits = recording_level_split(g, labels, seed=cfg["seed"])
    train_g = splits["train"] | splits["val"]; test_g = splits["test"]
    tr = np.isin(g, list(train_g))
    scaler = StandardScaler().fit(X[tr])
    head = OneVsRestClassifier(LogisticRegression(max_iter=3000, class_weight="balanced")).fit(scaler.transform(X[tr]), y[tr])
    head_classes = [classes[c] for c in head.classes_]

    man = pd.read_csv(cfg["paths"]["manifest"])
    file_for = {}
    for _, r in man.iterrows():
        gid = str(r["group_id"])
        if gid not in file_for:
            file_for[gid] = Path(cfg["paths"]["audio_dir"])/r["label"]/r["filename"]
    prim_by_species = defaultdict(list); any_by_species = defaultdict(list)
    for gid in set(g):
        sp = labels[g==gid][0]
        if gid in file_for:
            any_by_species[sp].append(gid)
            if gid in test_g: prim_by_species[sp].append(gid)
    sec_by_species = any_by_species if args.secondary_any else prim_by_species
    print("primary pool (test):", {k:len(v) for k,v in prim_by_species.items()})
    print("secondary pool:", {k:len(v) for k,v in sec_by_species.items()})

    embedder = BirdNETEmbedder(sample_rate=cfg["birdnet"]["sample_rate"])
    sr = cfg["birdnet"]["sample_rate"]; seg = sr*3
    rng = np.random.default_rng(cfg["seed"]); cache = {}
    def load_seg(gid):
        if gid not in cache:
            try:
                s,_ = librosa.load(str(file_for[gid]), sr=sr, mono=True); cache[gid]=energetic_segment(s,seg)
            except Exception: cache[gid]=None
        return cache[gid]

    species = [s for s in classes if prim_by_species.get(s)]
    for tau in args.taus:
        for snr in args.snrs:
            both=prim=sec=extra=total=0; sec_by=defaultdict(lambda:[0,0])
            for sA,sB in permutations(species,2):
                if not prim_by_species[sA] or not sec_by_species[sB]: continue
                for _ in range(args.n_per_pair):
                    a=load_seg(rng.choice(prim_by_species[sA])); b=load_seg(rng.choice(sec_by_species[sB]))
                    if a is None or b is None: continue
                    mix=mix_at_snr(a,b,snr)
                    wins=embedder.embed_signal(mix)
                    proba=head.predict_proba(scaler.transform(np.stack([w["embeddings"] for w in wins])))[0]
                    pred=predict_set(proba,head_classes,tau); total+=1
                    if sA in pred: prim+=1
                    if sB in pred: sec+=1; sec_by[sB][0]+=1
                    if sA in pred and sB in pred: both+=1
                    sec_by[sB][1]+=1
                    if pred-{sA,sB}: extra+=1
            if total:
                print(f"\n[tau={tau} SNR={snr:+.0f}dB] n={total}  both={both/total:.2f}  "
                      f"primary(louder)={prim/total:.2f}  secondary(quieter)={sec/total:.2f}  extra={extra/total:.2f}")
                print("  secondary detection by species:", {sp:f"{v[0]}/{v[1]}" for sp,v in sec_by.items()})

if __name__ == "__main__":
    main()
