import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score

N_SEEDS = 10
d = np.load("outputs/birdnet_embeddings.npz", allow_pickle=True)
X = d["X"].astype("float32"); y = d["y"]; g = np.array([str(x) for x in d["groups"]])
cls = [str(c) for c in d["classes"]]
present = [i for i in range(len(cls)) if (y == i).sum() > 0]

macro, acc = [], []; per_class = {i: [] for i in present}
for seed in range(N_SEEDS):
    oof = np.full(len(y), -1)
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(X,y,g):
        clf = make_pipeline(StandardScaler(),
              LogisticRegression(max_iter=3000,class_weight="balanced")).fit(X[tr],y[tr])
        oof[te] = clf.classes_[clf.predict_proba(X[te]).argmax(1)]
    macro.append(f1_score(y,oof,average="macro",labels=present,zero_division=0))
    acc.append(accuracy_score(y,oof))
    for i in present: per_class[i].append(f1_score(y==i,oof==i,zero_division=0))

def ms(v): return f"{np.mean(v):.3f} ± {np.std(v):.3f}"
print(f"=== {N_SEEDS}-seed x 5-fold recording-level CV (6 classes) ===")
print(f"macro-F1 : {ms(macro)}   (range {min(macro):.3f}-{max(macro):.3f})")
print(f"accuracy : {ms(acc)}")
print("\nper-class F1 (mean ± std over seeds):")
for i in present:
    print(f"  {cls[i]:24s} {ms(per_class[i])}   (n_rec={len(set(g[y==i]))})")
print("\nNote: eagle-owl CI is wide by construction (n_rec=7) — honest scarcity, not instability.")
