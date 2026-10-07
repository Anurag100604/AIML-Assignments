from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import matplotlib.pyplot as plt

# Load the real Iris dataset
data = load_iris(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for k in range(1, 16):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores.append(cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean())

best_k = int(np.argmax(scores) + 1)
print("Best k:", best_k)
print("Mean 5-fold CV accuracy:", scores[best_k - 1])

model = Pipeline([
    ("scale", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k)),
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Test accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# Figure 5.2: CV accuracy curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 16), scores, marker="o")
plt.axvline(best_k, linestyle="--", linewidth=1)
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Mean 5-Fold CV Accuracy")
plt.title("KNN Accuracy for Different Values of k")
plt.xticks(range(1, 16))
plt.grid(True)
plt.tight_layout()
plt.savefig("assignment5_accuracy_curve.png", dpi=200)
plt.show()

# Figure 5.3: confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot()
plt.title(f"KNN Confusion Matrix (k={best_k})")
plt.tight_layout()
plt.savefig("assignment5_confusion_matrix.png", dpi=200)
plt.show()

# Figure 5.4: decision regions with Petal Length and Petal Width only
X2 = X.iloc[:, [2, 3]].to_numpy()
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.20, random_state=42, stratify=y
)
region_model = Pipeline([
    ("scale", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k)),
]).fit(X2_train, y2_train)

x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
y_min, y_max = X2[:, 1].min() - 0.3, X2[:, 1].max() + 0.3
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.01),
)
Z = region_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, Z, alpha=0.25, cmap="viridis")
for class_id, name in enumerate(data.target_names):
    mask = y == class_id
    plt.scatter(X2[mask, 0], X2[mask, 1], label=name, s=45, edgecolors="black")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title(f"KNN Decision Regions using Petal Length and Petal Width (k={best_k})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("assignment5_decision_regions.png", dpi=200)
plt.show()
