import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.model_selection import StratifiedGroupKFold

d = np.load("outputs/birdnet_embeddings.npz", allow_pickle=True)
y = d["y"]; g = np.array([str(x) for x in d["groups"]]); X = d["X"]
cls = [str(c) for c in d["classes"]]
present = [i for i in range(len(cls)) if (y == i).sum() > 0]

print("=== recordings (groups) per class ===")
for i in present:
    print(f"  {cls[i]:24s} windows={int((y==i).sum()):4d}  recordings={len(set(g[y==i]))}")

bad = 0
for gid in set(g):
    if len(set(y[g == gid])) > 1:
        bad += 1; print(f"  ! group {gid} spans multiple labels")
print(f"\n[check] every recording has a single label: {'PASS' if bad==0 else 'FAIL'}")

seen_test = {}; leak = 0
for k,(tr,te) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=1337).split(X,y,g)):
    tr_g, te_g = set(g[tr]), set(g[te])
    if tr_g & te_g:
        leak += 1; print(f"  ! fold {k}: {len(tr_g & te_g)} groups in BOTH train and test")
    for gid in te_g: seen_test[gid] = seen_test.get(gid,0)+1
multi = {k:v for k,v in seen_test.items() if v>1}
print(f"[check] no train/test overlap within a fold: {'PASS' if leak==0 else 'FAIL'}")
print(f"[check] each recording tested exactly once: {'PASS' if not multi else 'FAIL'}")
print(f"[check] all recordings covered: {'PASS' if len(seen_test)==len(set(g)) else 'FAIL'} ({len(seen_test)}/{len(set(g))})")
if bad or leak or multi: sys.exit("LEAKAGE AUDIT FAILED")
print("\nAll leakage checks passed — recording-level splitting is clean.")
