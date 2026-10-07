from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt

# Load the real UCI Wine dataset
data = load_wine(as_frame=True)
X = StandardScaler().fit_transform(data.data)
y_true = data.target.to_numpy()

k_values = range(2, 9)
inertias = []
silhouettes = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X, labels))
    print(f"k={k}: inertia={model.inertia_:.3f}, silhouette={silhouettes[-1]:.6f}")

best_k = list(k_values)[int(np.argmax(silhouettes))]
final_model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
labels = final_model.fit_predict(X)

print("\nSelected k:", best_k)
print("Cluster sizes:", np.bincount(labels))
print("Adjusted Rand Index:", adjusted_rand_score(y_true, labels))

# Figure 2.2: Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (SSE)")
plt.title("K-Means Elbow Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("assignment2_elbow.png", dpi=200)
plt.show()

# Figure 2.3: Silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), silhouettes, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette Analysis")
plt.grid(True)
plt.tight_layout()
plt.savefig("assignment2_silhouette.png", dpi=200)
plt.show()

# Figure 2.4: PCA visualization of clusters
Z = PCA(n_components=2).fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=35)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"Wine Samples by K-Means Cluster (k={best_k})")
plt.grid(True)
plt.tight_layout()
plt.savefig("assignment2_kmeans_pca.png", dpi=200)
plt.show()

# Optional hierarchical clustering comparison from the practical record
for linkage in ["ward", "average", "complete"]:
    h_labels = AgglomerativeClustering(n_clusters=3, linkage=linkage).fit_predict(X)
    print(
        linkage,
        "silhouette=", round(silhouette_score(X, h_labels), 4),
        "ARI=", round(adjusted_rand_score(y_true, h_labels), 4),
        "sizes=", np.bincount(h_labels).tolist(),
    )
