"""
export_data.py
--------------
Runs the same pipeline as iris_classification.py (same seed, same split)
and exports every computed result to frontend/data.js, so the website
shows REAL numbers from scikit-learn (nothing is hardcoded).

Run:  python export_data.py
Then open frontend/index.html in your browser.
"""

import json
import os

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TEST_SIZE = 0.2
OUT = os.path.join("frontend", "data.js")

# ------------------------------------------------------------------ data
iris = load_iris(as_frame=True)
df = iris.frame.copy()
feat_cols = iris.feature_names
X, y = df[feat_cols], df["target"]
classes = list(iris.target_names)

features = [
    {"key": "sepal_length", "label": "Sepal length (cm)", "short": "Sepal length"},
    {"key": "sepal_width", "label": "Sepal width (cm)", "short": "Sepal width"},
    {"key": "petal_length", "label": "Petal length (cm)", "short": "Petal length"},
    {"key": "petal_width", "label": "Petal width (cm)", "short": "Petal width"},
]

rows = [[float(v) for v in X.iloc[i]] + [int(y.iloc[i])] for i in range(len(df))]

# per-class statistics
stats = {}
for c, name in enumerate(classes):
    sub = X[y == c]
    stats[name] = {
        "mean": [float(v) for v in sub.mean()],
        "std": [float(v) for v in sub.std()],
        "min": [float(v) for v in sub.min()],
        "max": [float(v) for v in sub.max()],
    }
overall = {
    "mean": [float(v) for v in X.mean()],
    "std": [float(v) for v in X.std()],
    "min": [float(v) for v in X.min()],
    "max": [float(v) for v in X.max()],
}
corr = [[float(v) for v in r] for r in X.corr().values]
F, p = f_classif(X, y)  # one-way ANOVA per feature
anova = [{"F": float(f), "p": float(pp)} for f, pp in zip(F, p)]
duplicates = int(df[feat_cols].duplicated().sum())

# ----------------------------------------------------------------- split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)
split = {
    "test_size": TEST_SIZE,
    "seed": SEED,
    "train_idx": [int(i) for i in X_train.index],
    "test_idx": [int(i) for i in X_test.index],
    "train_counts": [int((y_train == c).sum()) for c in range(3)],
    "test_counts": [int((y_test == c).sum()) for c in range(3)],
}

# ------------------------------------------------------------ preprocess
scaler = StandardScaler().fit(X_train)  # fit on TRAIN only
Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
scaler_json = {
    "mean": [float(v) for v in scaler.mean_],
    "scale": [float(v) for v in scaler.scale_],
}
train_range = {
    "min": [float(v) for v in X_train.min()],
    "max": [float(v) for v in X_train.max()],
}

# ----------------------------------------------------------------- train
models = {
    "lr": LogisticRegression(max_iter=200, random_state=SEED),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "dt": DecisionTreeClassifier(max_depth=3, random_state=SEED),
}
names = {"lr": "Logistic Regression", "knn": "K-Nearest Neighbors", "dt": "Decision Tree"}
for m in models.values():
    m.fit(Xtr, y_train)

# ------------------------------------------------------------- evaluate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results = {}
for key, m in models.items():
    pred = m.predict(Xte)
    pr, rc, f1, sup = precision_recall_fscore_support(y_test, pred, zero_division=0)
    cv_scores = cross_val_score(make_pipeline(StandardScaler(), m.__class__(**m.get_params())),
                                X, y, cv=cv, scoring="accuracy")
    results[key] = {
        "name": names[key],
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, average="macro")),
        "recall": float(recall_score(y_test, pred, average="macro")),
        "f1": float(f1_score(y_test, pred, average="macro")),
        "confusion": confusion_matrix(y_test, pred).tolist(),
        "per_class": [
            {"precision": float(pr[i]), "recall": float(rc[i]),
             "f1": float(f1[i]), "support": int(sup[i])} for i in range(3)
        ],
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "cv_folds": [float(s) for s in cv_scores],
        "test_pred": [int(v) for v in pred],
    }

# ------------------------------------------- model parameters for the JS
lr = models["lr"]
knn = models["knn"]
dt = models["dt"]


def export_tree(node=0):
    t = dt.tree_
    n = int(t.n_node_samples[node])
    counts = [int(round(v * n)) for v in t.value[node][0]]
    if t.children_left[node] == -1:
        return {"leaf": True, "cls": int(np.argmax(counts)), "samples": n, "counts": counts}
    f = int(t.feature[node])
    thr_cm = float(t.threshold[node] * scaler.scale_[f] + scaler.mean_[f])  # back to cm
    return {
        "leaf": False, "feature": f, "threshold": thr_cm, "samples": n, "counts": counts,
        "left": export_tree(int(t.children_left[node])),
        "right": export_tree(int(t.children_right[node])),
    }


payload = {
    "meta": {"sklearn": sklearn.__version__, "seed": SEED, "n": len(df)},
    "features": features,
    "classes": classes,
    "rows": rows,
    "stats": stats,
    "overall": overall,
    "corr": corr,
    "anova": anova,
    "duplicates": duplicates,
    "split": split,
    "scaler": scaler_json,
    "train_range": train_range,
    "results": results,
    "lr": {"coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist()},
    "knn": {"k": 5, "X": Xtr.tolist(), "y": [int(v) for v in y_train],
            "ids": [int(i) for i in X_train.index]},
    "tree": export_tree(),
}

os.makedirs("frontend", exist_ok=True)
with open(OUT, "w") as f:
    f.write("// AUTO-GENERATED by export_data.py - do not edit by hand.\n")
    f.write("window.IRIS = " + json.dumps(payload) + ";\n")
print(f"Wrote {OUT}")
for k, r in results.items():
    print(f"{r['name']:<22} test acc={r['accuracy']:.3f}  CV={r['cv_mean']:.3f} +/- {r['cv_std']:.3f}")
