import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load the real Engel household expenditure dataset
df = sm.datasets.engel.load_pandas().data
X = df[["income"]].values
y = df["foodexp"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

xs = StandardScaler().fit(X_train)
ys = StandardScaler().fit(y_train.reshape(-1, 1))

model = SVR(kernel="rbf", C=100, epsilon=0.1)
model.fit(
    xs.transform(X_train),
    ys.transform(y_train.reshape(-1, 1)).ravel(),
)

y_pred = ys.inverse_transform(
    model.predict(xs.transform(X_test)).reshape(-1, 1)
).ravel()

print("R2:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("MAE:", mean_absolute_error(y_test, y_pred))

# Figure 7.2: SVR fit
order = np.argsort(X_test[:, 0])
plt.figure(figsize=(8, 5))
plt.scatter(X_train[:, 0], y_train, s=25, label="Training data")
plt.scatter(X_test[:, 0], y_test, s=40, marker="x", label="Test data")
plt.plot(X_test[order, 0], y_pred[order], linewidth=2, label="SVR predictions")
plt.xlabel("Income")
plt.ylabel("Food expenditure")
plt.title("SVR Fit on Engel Data")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("assignment7_svr_fit.png", dpi=200)
plt.show()

# Figure 7.3: residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, s=40)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Predicted Food Expenditure")
plt.ylabel("Residual")
plt.title("SVR Residuals")
plt.grid(True)
plt.tight_layout()
plt.savefig("assignment7_residuals.png", dpi=200)
plt.show()

# Figure 7.4: sensitivity to C
C_values = [0.1, 1, 10, 100, 1000]
rmse_values = []
for C in C_values:
    svr = SVR(kernel="rbf", C=C, epsilon=0.1)
    svr.fit(xs.transform(X_train), ys.transform(y_train.reshape(-1, 1)).ravel())
    pred = ys.inverse_transform(
        svr.predict(xs.transform(X_test)).reshape(-1, 1)
    ).ravel()
    rmse_values.append(mean_squared_error(y_test, pred) ** 0.5)
    print(f"C={C}: RMSE={rmse_values[-1]:.4f}")

plt.figure(figsize=(8, 5))
plt.semilogx(C_values, rmse_values, marker="o")
plt.xlabel("C")
plt.ylabel("Test RMSE")
plt.title("Sensitivity of Test RMSE to C")
plt.grid(True)
plt.tight_layout()
plt.savefig("assignment7_c_sensitivity.png", dpi=200)
plt.show()
