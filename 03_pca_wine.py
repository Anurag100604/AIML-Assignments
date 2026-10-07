import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load and standardize the real Wine dataset
data = load_wine(as_frame=True)
X = StandardScaler().fit_transform(data.data)
y = data.target

pca = PCA()
Z = pca.fit_transform(X)
ev = pca.explained_variance_ratio_
cum = np.cumsum(ev)

print("Dataset shape:", data.data.shape)
print("\nExplained variance ratios:")
for i, value in enumerate(ev, start=1):
    print(f"PC{i}: {value:.6f} cumulative={cum[i-1]:.6f}")
print("\nFirst two PCs:", ev[:2].sum())

# Figure 3.2: Scree plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ev) + 1), ev, marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Scree Plot")
plt.grid(True)
plt.tight_layout()
plt.savefig("assignment3_scree.png", dpi=200)
plt.show()

# Figure 3.3: PC1 vs PC2 projection
Z2 = PCA(n_components=2).fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(Z2[:, 0], Z2[:, 1], c=y, s=35)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Wine Samples in PCA Space")
plt.grid(True)
plt.tight_layout()
plt.savefig("assignment3_pc1_pc2.png", dpi=200)
plt.show()

# Figure 3.4: eight largest absolute PC1 loadings
pc1 = pca.components_[0]
idx = np.argsort(np.abs(pc1))[-8:][::-1]
features = [data.feature_names[i] for i in idx]
values = [pc1[i] for i in idx]

plt.figure(figsize=(9, 5))
plt.barh(features[::-1], values[::-1])
plt.axvline(0, linewidth=1)
plt.xlabel("PC1 Loading")
plt.ylabel("Feature")
plt.title("Eight Largest Absolute Loadings on PC1")
plt.grid(axis="x")
plt.tight_layout()
plt.savefig("assignment3_pc1_loadings.png", dpi=200)
plt.show()
