import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load the real Grunfeld Investment dataset
df = sm.datasets.grunfeld.load_pandas().data
X = df[["value"]]
y = df["invest"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
print("R2:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("MAE:", mean_absolute_error(y_test, y_pred))

# Figure 6.2: actual versus predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, s=45)
lo = min(y_test.min(), y_pred.min())
hi = max(y_test.max(), y_pred.max())
plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
plt.xlabel("Actual Investment")
plt.ylabel("Predicted Investment")
plt.title("Actual vs Predicted Investment")
plt.grid(True)
plt.tight_layout()
plt.savefig("assignment6_actual_vs_predicted.png", dpi=200)
plt.show()

# Figure 6.3: residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, s=45)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Predicted Investment")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.grid(True)
plt.tight_layout()
plt.savefig("assignment6_residuals.png", dpi=200)
plt.show()

# Figure 6.4: slope coefficient
plt.figure(figsize=(6, 5))
plt.bar(["Market Value"], [model.coef_[0]])
plt.axhline(0, linewidth=1)
plt.ylabel("Estimated Slope (beta1)")
plt.title("Estimated Slope Coefficient for Market Value")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("assignment6_slope.png", dpi=200)
plt.show()
