import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

d = np.load("outputs/birdnet_embeddings.npz", allow_pickle=True)
X = d["X"].astype("float32"); y = d["y"]; g = np.array([str(x) for x in d["groups"]])
cls = [str(c) for c in d["classes"]]
present = [i for i in range(len(cls)) if (y == i).sum() > 0]
owl = cls.index("indian_eagle_owl")

def l2(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

def proto_predict(Xtr, ytr, Xte, metric, normed):
    if normed:
        Xtr, Xte = l2(Xtr), l2(Xte)
    protos = {c: Xtr[ytr == c].mean(0) for c in present}
    cs = np.array(present); M = np.stack([protos[c] for c in cs])
    if metric == "cosine":
        S = l2(Xte) @ l2(M).T          # higher = closer
        return cs[S.argmax(1)]
    else:                               # euclidean
        D = ((Xte[:, None, :] - M[None, :, :]) ** 2).sum(-1)
        return cs[D.argmin(1)]

configs = [("LR baseline", None, None),
           ("proto-cosine",      "cosine",    False),
           ("proto-cos-L2norm",  "cosine",    True),
           ("proto-euclid",      "euclidean", False),
           ("proto-euclid-std",  "euclidean", True)]

print(f"{'method':20s}{'macroF1':>9}{'owlF1':>8}{'owlRecall':>11}")
for name, metric, normed in configs:
    oof = np.full(len(y), -1)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=1337).split(X, y, g):
        if metric is None:
            clf = make_pipeline(StandardScaler(),
                  LogisticRegression(max_iter=3000, class_weight="balanced")).fit(X[tr], y[tr])
            oof[te] = clf.predict(X[te])
        else:
            sc = StandardScaler().fit(X[tr])
            oof[te] = proto_predict(sc.transform(X[tr]), y[tr], sc.transform(X[te]), metric, normed)
    macro = f1_score(y, oof, average="macro", labels=present, zero_division=0)
    owlf1 = f1_score(y == owl, oof == owl, zero_division=0)
    owlr = (oof[y == owl] == owl).mean()
    print(f"{name:20s}{macro:9.3f}{owlf1:8.3f}{owlr:11.3f}")
