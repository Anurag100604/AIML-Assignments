from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the real scikit-learn Diabetes dataset
data = load_diabetes(as_frame=True)
df = data.frame.copy()

X = df.drop(columns=["target"])
y = df["target"]

print(df.head(10))
print("\nShape:", df.shape)

# Pearson associations
corr = X.corrwith(y).sort_values()
print("\nPearson correlation with target:")
print(corr)

print("\nBMI Pearson r:", df["bmi"].corr(y))
print("BMI Spearman rho:", df["bmi"].corr(y, method="spearman"))

# Figure 1.2: feature-target correlations
plt.figure(figsize=(9, 5))
corr.plot(kind="barh")
plt.axvline(0, linewidth=1)
plt.xlabel("Pearson correlation with target")
plt.title("Diabetes Feature-Target Correlations")
plt.tight_layout()
plt.savefig("assignment1_correlation_bar.png", dpi=200)
plt.show()

# Figure 1.3: BMI scatter with straight trend line
x = df["bmi"].to_numpy()
slope, intercept = np.polyfit(x, y.to_numpy(), 1)
x_line = np.linspace(x.min(), x.max(), 200)
y_line = slope * x_line + intercept

plt.figure(figsize=(8, 5))
plt.scatter(x, y, s=18, label="Observed observations")
plt.plot(x_line, y_line, linewidth=2, label=f"Trend line (r={df['bmi'].corr(y):.3f})")
plt.xlabel("BMI (scaled)")
plt.ylabel("Disease progression")
plt.title(f"BMI vs Diabetes Progression (r={df['bmi'].corr(y):.3f})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("assignment1_bmi_scatter.png", dpi=200)
plt.show()

# Full correlation heatmap
corr_all = df.corr(numeric_only=True)
plt.figure(figsize=(9, 7))
plt.imshow(corr_all, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr_all.columns)), corr_all.columns, rotation=45, ha="right")
plt.yticks(range(len(corr_all.columns)), corr_all.columns)
plt.title("Full Correlation Heatmap")
plt.tight_layout()
plt.savefig("assignment1_heatmap.png", dpi=200)
plt.show()
