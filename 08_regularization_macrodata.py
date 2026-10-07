import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load real macroeconomic data
df = sm.datasets.macrodata.load_pandas().data
features = ["realcons", "realinv", "realgovt", "realdpi", "cpi", "realint"]
X, y = df[features], df["realgdp"]

# Preserve chronological order; last 25% is the final test segment
split = int(len(df) * 0.75)
X_tr, X_te = X.iloc[:split], X.iloc[split:]
y_tr, y_te = y.iloc[:split], y.iloc[split:]

alphas = np.logspace(-3, 3, 13)
makers = {
    "Ridge": lambda a: Ridge(alpha=a),
    "Lasso": lambda a: Lasso(alpha=a, max_iter=200000),
    "ElasticNet": lambda a: ElasticNet(alpha=a, l1_ratio=0.5, max_iter=200000),
}

best = {}
cv_curves = {}
tscv = TimeSeriesSplit(n_splits=5)

for name, make in makers.items():
    cv_rmse = []
    for a in alphas:
        fold_errors = []
        for tr, va in tscv.split(X_tr):
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("model", make(a)),
            ])
            pipe.fit(X_tr.iloc[tr], y_tr.iloc[tr])
            pred = pipe.predict(X_tr.iloc[va])
            fold_errors.append(mean_squared_error(y_tr.iloc[va], pred) ** 0.5)
        cv_rmse.append(np.mean(fold_errors))
    cv_curves[name] = cv_rmse
    best[name] = float(alphas[int(np.argmin(cv_rmse))])

print("Selected alpha:", best)

models = {
    "Linear": Pipeline([
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ])
}
models.update({
    n: Pipeline([
        ("scale", StandardScaler()),
        ("model", makers[n](best[n])),
    ])
    for n in makers
})

results = {}
predictions = {}
coefficients = {}

for name, model in models.items():
    model.fit(X_tr, y_tr)
    train_pred = model.predict(X_tr)
    test_pred = model.predict(X_te)
    predictions[name] = test_pred
    reg = model.named_steps["model"]
    coefficients[name] = reg.coef_
    results[name] = {
        "train_r2": r2_score(y_tr, train_pred),
        "test_r2": r2_score(y_te, test_pred),
        "rmse": mean_squared_error(y_te, test_pred) ** 0.5,
        "mae": mean_absolute_error(y_te, test_pred),
    }
    print(
        name,
        "train R2 %.4f" % results[name]["train_r2"],
        "test R2 %.4f" % results[name]["test_r2"],
        "RMSE %.1f" % results[name]["rmse"],
        "MAE %.1f" % results[name]["mae"],
    )

# Figure 8.2: validation RMSE vs alpha
plt.figure(figsize=(9, 5))
for name, curve in cv_curves.items():
    plt.semilogx(alphas, curve, marker="o", label=name)
plt.xlabel("Alpha")
plt.ylabel("Mean Time-Series CV RMSE")
plt.title("Validation RMSE vs Regularization Strength")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("assignment8_validation_rmse_vs_alpha.png", dpi=200)
plt.show()

# Figure 8.3: test RMSE comparison
labels = list(results)
test_rmse = [results[n]["rmse"] for n in labels]
plt.figure(figsize=(8, 5))
plt.bar(labels, test_rmse)
plt.ylabel("Test RMSE")
plt.title("Test RMSE Comparison")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("assignment8_test_rmse_comparison.png", dpi=200)
plt.show()

# Figure 8.4: standardized coefficients
coef_df = pd.DataFrame(coefficients, index=features)
plt.figure(figsize=(11, 6))
coef_df.plot(kind="bar", ax=plt.gca())
plt.axhline(0, linewidth=1)
plt.xlabel("Feature")
plt.ylabel("Coefficient on Standardized Predictor")
plt.title("Coefficients of the Four Regression Models")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("assignment8_coefficients.png", dpi=200)
plt.show()
