import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, recall_score, precision_score

d = np.load("outputs/birdnet_embeddings.npz", allow_pickle=True)
X = d["X"].astype("float32"); y = d["y"]; g = np.array([str(x) for x in d["groups"]])
cls = [str(c) for c in d["classes"]]
present = [i for i in range(len(cls)) if (y == i).sum() > 0]
owl = cls.index("indian_eagle_owl"); cuc = cls.index("indian_cuckoo")

P = np.zeros((len(y), len(cls)), dtype=float)
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=1337).split(X, y, g):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, class_weight="balanced")).fit(X[tr], y[tr])
    proba = clf.predict_proba(X[te])           # columns = clf.classes_ order
    for j, c in enumerate(clf.classes_):       # write each class column explicitly
        P[te, c] = proba[:, j]

# sanity: argmax should reproduce the baseline (~0.55 owl recall, ~0.86 macro)
base_pred = P.argmax(1)
print("SANITY argmax: owl_recall=%.3f  macroF1=%.3f  (expect ~0.55 / ~0.86)" %
      (recall_score(y == owl, base_pred == owl, zero_division=0),
       f1_score(y, base_pred, average="macro", labels=present, zero_division=0)))
print()
print(f"{'owl_thr':>8}{'owlRecall':>11}{'owlPrec':>9}{'owlF1':>8}{'cucPrec':>9}{'macroF1':>9}")
owl_p_col = P[:, owl]
for t in [None, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]:
    pred = P.argmax(1).copy()
    if t is not None:
        pred[owl_p_col >= t] = owl
    owl_r = recall_score(y == owl, pred == owl, zero_division=0)
    owl_pr = precision_score(y == owl, pred == owl, zero_division=0)
    owl_f = f1_score(y == owl, pred == owl, zero_division=0)
    cuc_p = precision_score(y == cuc, pred == cuc, zero_division=0)
    macro = f1_score(y, pred, average="macro", labels=present, zero_division=0)
    lab = "argmax" if t is None else f"{t:.2f}"
    print(f"{lab:>8}{owl_r:11.3f}{owl_pr:9.3f}{owl_f:8.3f}{cuc_p:9.3f}{macro:9.3f}")
