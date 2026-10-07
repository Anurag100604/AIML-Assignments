import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Load the real Breast Cancer Wisconsin (Diagnostic) dataset
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = GaussianNB().fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion matrix:\n", cm)
print("\nClassification report:\n", classification_report(
    y_test, y_pred, target_names=data.target_names
))

# Figure 4.2: confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot()
plt.title("Gaussian Naive Bayes Confusion Matrix")
plt.tight_layout()
plt.savefig("assignment4_confusion_matrix.png", dpi=200)
plt.show()

# Figure 4.3: class distribution
class_counts = pd.Series(y).map({0: data.target_names[0], 1: data.target_names[1]}).value_counts()
plt.figure(figsize=(7, 5))
class_counts.plot(kind="bar")
plt.xlabel("Class")
plt.ylabel("Number of observations")
plt.title("Real Class Distribution")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("assignment4_class_distribution.png", dpi=200)
plt.show()

# Figure 4.4: predicted probability distribution for benign class
plt.figure(figsize=(8, 5))
plt.hist(y_prob[:, 1], bins=15, edgecolor="black")
plt.xlabel("Predicted probability of benign class")
plt.ylabel("Number of test observations")
plt.title("Test-Set Predicted Probability Distribution for the Benign Class")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("assignment4_probability_distribution.png", dpi=200)
plt.show()
