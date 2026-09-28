import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

d = np.load("outputs/birdnet_embeddings.npz", allow_pickle=True)
X = d["X"].astype("float32"); y = d["y"]; g = np.array([str(x) for x in d["groups"]])
cls = [str(c) for c in d["classes"]]
present = [i for i in range(len(cls)) if (y == i).sum() > 0]
owl = cls.index("indian_eagle_owl")

def make(name):
    if name == "LR (baseline)":
        return LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    if name == "SVM-RBF":
        return SVC(kernel="rbf", class_weight="balanced", C=10, gamma="scale")
    if name == "kNN(5)":
        return KNeighborsClassifier(n_neighbors=5, weights="distance")
    if name == "MLP(256)":
        return MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, alpha=1e-3)

heads = ["LR (baseline)", "SVM-RBF", "kNN(5)", "MLP(256)"]
print(f"{'head':16s}{'macroF1':>10}{'owlF1':>8}")
for name in heads:
    oof = np.full(len(y), -1)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=1337).split(X, y, g):
        clf = make_pipeline(StandardScaler(), make(name)).fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
    macro = f1_score(y, oof, average="macro", labels=present, zero_division=0)
    owlf1 = f1_score(y == owl, oof == owl, zero_division=0)
    print(f"{name:16s}{macro:10.3f}{owlf1:8.3f}")
