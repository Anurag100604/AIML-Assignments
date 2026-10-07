import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Required real MediaEval 2016 development-set files
REQUIRED = ["posts.txt", "adjpg.csv", "nadjpg.csv", "sadjpg.csv", "snadjpg.csv"]
for filename in REQUIRED:
    if not Path(filename).exists():
        raise FileNotFoundError(f"Missing required file: {filename}")

posts = pd.read_csv("posts.txt", sep="\t", quoting=csv.QUOTE_NONE, dtype=str)


def read_matrix_file(name):
    out = {}
    with open(name + ".csv", encoding="utf-8", errors="ignore") as file:
        for line in file:
            p = [x.strip() for x in line.split(",")]
            if len(p) > 3:
                try:
                    out[p[0]] = (int(p[1]), int(p[2]), np.array(p[3:], dtype=float))
                except ValueError:
                    pass
    return out


def stats(v):
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return [v.mean(), v.std(), v.min(), v.max(), *q, (v > 0).mean()]


F = {n: read_matrix_file(n) for n in ["adjpg", "nadjpg", "sadjpg", "snadjpg"]}
ids = set.intersection(*[set(F[n]) for n in F])

img_feat = {}
for image_id in ids:
    img_feat[image_id] = (
        stats(F["adjpg"][image_id][2])
        + stats(F["nadjpg"][image_id][2])
        + list(F["sadjpg"][image_id][2])
        + list(F["snadjpg"][image_id][2])
        + [F["adjpg"][image_id][0], F["adjpg"][image_id][1]]
    )

posts["img"] = posts["image_id(s)"].apply(
    lambda s: next((x.strip() for x in str(s).split(",") if x.strip() in img_feat), None)
)
d = posts.dropna(subset=["img"]).copy()

d["text"] = (
    d["post_text"].fillna("")
    .str.replace(r"http\S+", "URL", regex=True)
    .str.replace(r"@\w+", "@USER", regex=True)
)

y = (d["label"].str.lower() == "fake").astype(int).to_numpy()
X_img = np.nan_to_num(np.array([img_feat[i] for i in d["img"]]))

prob_text = np.zeros(len(d))
prob_img = np.zeros(len(d))
prob_fused = np.zeros(len(d))

for tr, te in GroupKFold(n_splits=5).split(d, y, d["img"]):
    tv = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True
    )
    T_tr = tv.fit_transform(d.text.values[tr])
    T_te = tv.transform(d.text.values[te])

    sc = StandardScaler().fit(X_img[tr])
    I_tr = np.clip(sc.transform(X_img[tr]), -5, 5)
    I_te = np.clip(sc.transform(X_img[te]), -5, 5)

    text_clf = LogisticRegression(C=1.0, max_iter=3000).fit(T_tr, y[tr])
    img_clf = LogisticRegression(C=0.3, max_iter=3000).fit(I_tr, y[tr])
    fused_clf = LogisticRegression(C=1.0, max_iter=3000).fit(
        hstack([T_tr, csr_matrix(I_tr) * 0.3]).tocsr(), y[tr]
    )

    prob_text[te] = text_clf.predict_proba(T_te)[:, 1]
    prob_img[te] = img_clf.predict_proba(I_te)[:, 1]
    prob_fused[te] = fused_clf.predict_proba(
        hstack([T_te, csr_matrix(I_te) * 0.3]).tocsr()
    )[:, 1]

prob_late = (prob_text + prob_img) / 2


def metrics(prob):
    pred = (prob >= 0.5).astype(int)
    return (
        accuracy_score(y, pred),
        f1_score(y, pred, average="macro"),
        precision_score(y, pred, zero_division=0),
        recall_score(y, pred, zero_division=0),
        roc_auc_score(y, prob),
    )

print("Text only:", metrics(prob_text))
print("Image only:", metrics(prob_img))
print("Early fusion:", metrics(prob_fused))
print("Late fusion:", metrics(prob_late))

# Majority baseline is always the majority class; ROC-AUC is 0.5 for a constant score.
majority_positive = float((y == 1).mean())
majority_pred = np.full(len(y), int(majority_positive >= 0.5))
print("Majority baseline accuracy:", accuracy_score(y, majority_pred))

# Figure 10.2: the same three model groups shown in the practical record
model_names = ["Text only", "Image forensics only", "Early fusion (text + image)"]
probabilities = [prob_text, prob_img, prob_fused]
metric_rows = [metrics(p) for p in probabilities]

accuracy = [row[0] for row in metric_rows]
macro_f1 = [row[1] for row in metric_rows]
roc_auc = [row[4] for row in metric_rows]

x = np.arange(len(model_names))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, accuracy, width, label="Accuracy")
plt.bar(x, macro_f1, width, label="Macro-F1")
plt.bar(x + width, roc_auc, width, label="ROC-AUC")
plt.axhline(majority_positive, linestyle="--", linewidth=1, label=f"Majority baseline = {majority_positive:.3f}")
plt.xticks(x, model_names)
plt.ylabel("Score")
plt.title(f"Image-grouped 5-fold cross-validation ({len(d):,} posts)")
plt.ylim(0.4, 0.95)
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("assignment10_model_comparison.png", dpi=200)
plt.show()

# Figure 10.3: early-fusion confusion matrix from the actual code execution
fusion_pred = (prob_fused >= 0.5).astype(int)
cm = confusion_matrix(y, fusion_pred)
print("Fusion confusion matrix:\n", cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm, aspect="auto")
plt.colorbar(label="Count")
plt.xticks([0, 1], ["real", "fake"])
plt.yticks([0, 1], ["real", "fake"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Fusion model: out-of-fold posts")
plt.tight_layout()
plt.savefig("assignment10_confusion_matrix.png", dpi=200)
plt.show()
