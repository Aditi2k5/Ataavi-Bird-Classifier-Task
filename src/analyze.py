import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, precision_score, recall_score

NPZ = "outputs/birdnet_embeddings.npz"
d = np.load(NPZ, allow_pickle=True)
X = d["X"].astype(np.float32); y = d["y"]
groups = np.array([str(g) for g in d["groups"]])
classes = [str(c) for c in d["classes"]]
faint = d["is_faint"].astype(bool); overlap = d["has_overlap"].astype(bool)

# drop classes with zero windows (e.g. excluded magpie-robin) from reporting
present = [i for i in range(len(classes)) if (y == i).sum() > 0]
present_names = [classes[i] for i in present]
owl = classes.index("indian_eagle_owl")

print("=== dataset ===")
for i in present:
    n_rec = len(set(groups[y == i]))
    print(f"  {classes[i]:24s} {int((y==i).sum()):4d} windows / {n_rec} recordings")

# ---- out-of-fold predictions + per-fold owl detail ----
oof = np.full(len(y), -1)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1337)
print("\n=== per fold ===")
for k, (tr, te) in enumerate(sgkf.split(X, y, groups)):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0))
    clf.fit(X[tr], y[tr])
    pred = clf.classes_[clf.predict_proba(X[te]).argmax(1)]
    oof[te] = pred
    macro = f1_score(y[te], pred, average="macro", labels=present, zero_division=0)
    owl_recs = sorted(set(groups[te][y[te] == owl]))
    m = y[te] == owl
    if m.sum():
        owl_f1 = f1_score(y[te]==owl, pred==owl, zero_division=0)
        owl_p = precision_score(y[te]==owl, pred==owl, zero_division=0)
        owl_r = recall_score(y[te]==owl, pred==owl, zero_division=0)
        # what's hard about these owl recordings?
        faint_frac = faint[te][m].mean(); ov_frac = overlap[te][m].mean()
        print(f"fold {k}: macroF1={macro:.3f} | owl F1={owl_f1:.3f} "
              f"P={owl_p:.2f} R={owl_r:.2f} | owl recs={owl_recs} "
              f"(faint {faint_frac:.0%}, overlap {ov_frac:.0%})")
    else:
        print(f"fold {k}: macroF1={macro:.3f} | NO owl in test | owl recs=[]")

# ---- per-class F1 within slices ----
def per_class_in_slice(mask, name):
    print(f"\n=== {name} slice (n={int(mask.sum())}) — per-class F1 ===")
    yt, yp = y[mask], oof[mask]
    for i in present:
        support = int((yt == i).sum())
        if support == 0:
            print(f"  {classes[i]:24s}   (no examples in slice)"); continue
        f1 = f1_score(yt == i, yp == i, zero_division=0)
        print(f"  {classes[i]:24s} F1={f1:.3f}  (n={support})")
    print(f"  {'OVERALL macro':24s} F1={f1_score(yt, yp, average='macro', labels=present, zero_division=0):.3f}")

full_macro = f1_score(y, oof, average="macro", labels=present, zero_division=0)
print(f"\n=== overall (pooled OOF) macro-F1 = {full_macro:.3f} ===")
per_class_in_slice(~faint & ~overlap, "CLEAN (not faint, not overlap)")
per_class_in_slice(faint, "FAINT")
per_class_in_slice(overlap, "OVERLAP")
