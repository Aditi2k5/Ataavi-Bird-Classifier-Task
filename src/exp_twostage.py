import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

d = np.load("outputs/birdnet_embeddings.npz", allow_pickle=True)
X = d["X"].astype("float32"); y = d["y"]; g = np.array([str(x) for x in d["groups"]])
cls = [str(c) for c in d["classes"]]
present = [i for i in range(len(cls)) if (y == i).sum() > 0]
owl = cls.index("indian_eagle_owl"); cuc = cls.index("indian_cuckoo")

def l2(a): return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

def run(bin_thresh):
    oof = np.full(len(y), -1)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=1337).split(X, y, g):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        # stage 1: multiclass LR
        lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, y[tr])
        pred = lr.classes_[lr.predict_proba(Xte).argmax(1)]
        # stage 2a: prototype candidate = cosine-nearest to owl among {owl,cuckoo}
        owl_proto = l2(Xtr[y[tr] == owl]).mean(0); cuc_proto = l2(Xtr[y[tr] == cuc]).mean(0)
        Xn = l2(Xte)
        cand = (Xn @ owl_proto) > (Xn @ cuc_proto)      # closer to owl than cuckoo
        # stage 2b: dedicated owl-vs-cuckoo LR confirms
        m = (y[tr] == owl) | (y[tr] == cuc)
        b = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr[m], (y[tr][m] == owl).astype(int))
        confirm = b.predict_proba(Xte)[:, 1] >= bin_thresh
        # rescue: only relabel windows currently called cuckoo, that are candidates AND confirmed
        rescue = (pred == cuc) & cand & confirm
        pred[rescue] = owl
        oof[te] = pred
    return oof

print(f"{'config':22s}{'macroF1':>9}{'owlF1':>8}{'owlP':>7}{'owlR':>7}{'cucP':>7}")
# baseline for reference
base = np.full(len(y), -1)
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=1337).split(X, y, g):
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(X[tr], y[tr])
    base[te] = lr.predict(X[te])
def report(name, oof):
    print(f"{name:22s}{f1_score(y,oof,average='macro',labels=present,zero_division=0):9.3f}"
          f"{f1_score(y==owl,oof==owl,zero_division=0):8.3f}"
          f"{precision_score(y==owl,oof==owl,zero_division=0):7.3f}"
          f"{recall_score(y==owl,oof==owl,zero_division=0):7.3f}"
          f"{precision_score(y==cuc,oof==cuc,zero_division=0):7.3f}")
report("LR baseline", base)
for t in [0.5, 0.6, 0.7, 0.8]:
    report(f"two-stage (bin>={t})", run(t))
