import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
d=np.load("outputs/birdnet_embeddings.npz",allow_pickle=True)
X=d["X"].astype("float32");y=d["y"];g=np.array([str(x) for x in d["groups"]])
cls=[str(c) for c in d["classes"]];present=[i for i in range(len(cls)) if (y==i).sum()>0]
owl=cls.index("indian_eagle_owl")
# out-of-fold owl probability
Pown=np.zeros(len(y))
for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=1337).split(X,y,g):
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced")).fit(X[tr],y[tr])
    pr=clf.predict_proba(X[te]); j=list(clf.classes_).index(owl); Pown[te]=pr[:,j]
yt=(y==owl).astype(int)
n_owl=int(yt.sum())
print(f"owl windows={n_owl}  (7 recordings)\n")
print(f"{'thr':>6}{'precision':>11}{'recall':>9}{'F1':>7}{'TP':>5}{'FP':>5}  (of {n_owl} owls)")
for t in [0.5,0.6,0.7,0.8,0.9,0.95]:
    pred=(Pown>=t).astype(int)
    tp=int((pred[yt==1]==1).sum()); fp=int((pred[yt==0]==1).sum())
    print(f"{t:>6.2f}{precision_score(yt,pred,zero_division=0):>11.3f}"
          f"{recall_score(yt,pred,zero_division=0):>9.3f}{f1_score(yt,pred,zero_division=0):>7.3f}{tp:>5}{fp:>5}")
print("\nRead: pick the row where precision meets your bar; recall is the coverage you keep.")
