from __future__ import annotations
import argparse, csv, glob, hashlib, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config

def group_id_for(stem): return stem.split("__")[0]
AUDIO_EXTS = ("*.wav","*.mp3","*.flac","*.ogg","*.WAV","*.MP3")
def list_species_files(folder):
    files=[]
    for pat in AUDIO_EXTS: files.extend(folder.glob(pat))
    return sorted(set(files))
def _sig_hash(sig):
    a=np.ascontiguousarray(sig[:48000].astype(np.float32))
    return hashlib.md5(a.tobytes()).hexdigest()+f"_{len(sig)}"
def load_overlap_map(cfg):
    man=Path(cfg["paths"]["manifest"]); m={}
    if man.exists():
        for row in csv.DictReader(open(man)):
            if "xc_id" in row and "has_overlap" in row:
                m[str(row["xc_id"])]=str(row["has_overlap"]).strip().lower()=="true"
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="config.yaml")
    ap.add_argument("--list-only",action="store_true")
    args=ap.parse_args()
    cfg=load_config(args.config)
    classes=[s["label"] for s in cfg["species"]]
    audio_dir=Path(cfg["paths"]["audio_dir"])
    overlap_map=load_overlap_map(cfg)
    import librosa
    sr=cfg["birdnet"]["sample_rate"]

    if args.list_only:
        for label in classes:
            files=list_species_files(audio_dir/label); seen=set(); groups={}; dups=0
            for f in files:
                try: sig,_=librosa.load(str(f),sr=sr,mono=True)
                except Exception as e: print(f"  ! {f.name}: {str(e)[:50]}"); continue
                h=_sig_hash(sig)
                if h in seen: dups+=1; continue
                seen.add(h); groups.setdefault(group_id_for(f.stem),[]).append(f.name)
            print(f"{label:24s} {len(files)} files -> {len(groups)} recordings ({dups} dup skipped)")
        return

    from baseline_birdnet import BirdNETEmbedder
    embedder=BirdNETEmbedder(sample_rate=sr)
    X,y,groups,rms_vals,overlap,overlap_known=[],[],[],[],[],[]; man_rows=[]
    for ci,label in enumerate(classes):
        files=list_species_files(audio_dir/label); seen=set(); n_ok=0
        for f in files:
            try: sig,_=librosa.load(str(f),sr=sr,mono=True)
            except Exception as e: print(f"  ! load fail {f.name}: {str(e)[:50]}"); continue
            h=_sig_hash(sig)
            if h in seen: continue
            seen.add(h); gid=group_id_for(f.stem)
            known=gid in overlap_map; ov=overlap_map.get(gid,False)
            for w in embedder.embed_signal(sig):
                X.append(w["embeddings"]); y.append(ci); groups.append(gid)
                rms_vals.append(w["rms"]); overlap.append(ov); overlap_known.append(known)
            n_ok+=1
            man_rows.append({"label":label,"group_id":gid,"filename":f.name,
                "source":"xeno-canto" if known else "team-provided",
                "has_overlap":ov,"overlap_known":known})
        n_rec=len({r["group_id"] for r in man_rows if r["label"]==label})
        print(f"  {label:24s} {n_ok} files -> {n_rec} recordings")
    rms_vals=np.asarray(rms_vals,float)
    thr=np.nanpercentile(rms_vals,cfg["birdnet"]["faint_rms_percentile"])
    out={"X":np.stack(X),"y":np.asarray(y),"groups":np.asarray(groups),
         "is_faint":(rms_vals<=thr),"has_overlap":np.asarray(overlap,bool),
         "overlap_known":np.asarray(overlap_known,bool),"classes":np.asarray(classes)}
    dest=Path(cfg["paths"]["out_dir"])/"birdnet_embeddings.npz"; dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,**out)
    with open(cfg["paths"]["manifest"],"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(man_rows[0].keys())); w.writeheader(); w.writerows(man_rows)
    print(f"\n[cached] {dest}  X={out['X'].shape}")
    print(f"  faint={int(out['is_faint'].sum())}  overlap(known)={int((out['has_overlap']&out['overlap_known']).sum())}")
    print("  recordings/class:",{c:len({r['group_id'] for r in man_rows if r['label']==c}) for c in classes})

if __name__=="__main__": main()
