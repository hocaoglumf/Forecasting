import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# 1) Example time series (monthly demand)
#    (trend + seasonality + noise)
# -------------------------------
np.random.seed(7)

n_months = 36
t = np.arange(1, n_months + 1)

trend = 2.0 * t
season = 20.0 * np.sin(2 * np.pi * t / 12)   # yearly seasonality
noise = np.random.normal(0, 5, size=n_months)

y = 200 + trend + season + noise  # demand series

# -------------------------------
# 2) Build supervised dataset with lags
# -------------------------------
lags = [1, 2, 3, 12]  # common choice for monthly data
rows = []

for i in range(max(lags), len(y)):
    row = {f"lag_{l}": y[i - l] for l in lags}
    row["month"] = (i % 12) + 1   # 1..12 (helps seasonality)
    row["t"] = i + 1              # time index (month number)
    row["y"] = y[i]               # target
    rows.append(row)

df = pd.DataFrame(rows)

# -------------------------------
# 3) Time-based train/test split
#    Train: t <= 30, Test: t > 30
# -------------------------------
train = df[df["t"] <= 30].copy()
test  = df[df["t"] > 30].copy()

X_train = train.drop(columns=["y"])
y_train = train["y"]

X_test = test.drop(columns=["y"])
y_test = test["y"]

# -------------------------------
# 4) Train Random Forest
# -------------------------------
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=7,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# -------------------------------
# 5) Evaluate on test set
# -------------------------------
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)

print("Test MAE :", round(mae, 3))
print("Test RMSE:", round(rmse, 3))

# Show actual vs predicted for the test months
results = pd.DataFrame({
    "t": test["t"].values,
    "actual": y_test.values,
    "predicted": y_pred
})
print("The Data: ", y)
print("\nActual vs Predicted (Test):")
print(results.to_string(index=False))

# -------------------------------
# 6) One-step-ahead forecast (Month 37)
# -------------------------------
# Features for t = 37 use the latest observed lags from y[0..35]
i = 36  # 0-based index for month 37 (just for feature construction)

x_next = {f"lag_{l}": y[(i - 1) - (l - 1)] for l in lags}  # uses y[35], y[34], ...
# Explanation:
# lag_1 should be y[36-1] = y[35]
# lag_2 should be y[34]
# lag_3 should be y[33]
# lag_12 should be y[24]

x_next["month"] = (i % 12) + 1
x_next["t"] = i + 1

y_next = rf.predict(pd.DataFrame([x_next]))[0]
print("\nForecast for month 37:", round(float(y_next), 3))