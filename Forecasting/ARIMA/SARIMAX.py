import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX

# ----------------------------
# Data
# ----------------------------
y = pd.Series([
    280.05, 305.18, 324.14, 355.64, 394.53, 438.96, 501.13, 572.49, 635.81,
    726.68, 784.94, 727.34, 713.08, 747.34, 771.44, 894.97, 894.46, 968.30,
    1012.44, 1046.07, 1132.78, 1252.69, 1187.70, 1205.45, 1241.29, 1319.69,
    1426.17, 1856.83, 2023.56, 2129.56, 2239.85, 2389.03, 2660.00, 2770.00,
    2863.00, 3228.00, 3557.00, 3894.00, 4171.00, 4580.00, 4644.00, 4621.00,
    5142.00, 6440.00, 6448.00, 5960.00, 6155.00, 6507.00, 6001.00, 5927.00,
    6731.00, 7614.00, 7103.00, 6746.00, 7328.00, 8212.00, 7993.00, 8300.00,
    8731.00, 8702.00, 6677.00
], dtype=float)

idx = pd.period_range("2021-01", periods=len(y), freq="M")
y.index = idx

# ----------------------------
# Train/Test
# ----------------------------
TEST_SIZE = 12
y_train, y_test = y.iloc[:-TEST_SIZE], y.iloc[-TEST_SIZE:]

# ----------------------------
# SARIMAX config
# order=(p,d,q), seasonal_order=(P,D,Q,s)
# Typical starter for monthly: (1,1,1) + seasonal(1,1,1,12)
# ----------------------------
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)

model = SARIMAX(
    y_train,
    order=order,
    seasonal_order=seasonal_order,
    trend="n",                 # "n" none, "c" constant, "t" trend, "ct" both
    enforce_stationarity=False,
    enforce_invertibility=False
)

fit = model.fit(disp=False)
print(fit.summary())

# ----------------------------
# Test prediction
# ----------------------------
pred_test = fit.get_forecast(steps=TEST_SIZE)
yhat_test = pred_test.predicted_mean
ci_test = pred_test.conf_int()

# ----------------------------
# Future forecast
# ----------------------------
H = 12
pred_future = fit.get_forecast(steps=TEST_SIZE + H)
yhat_all = pred_future.predicted_mean
ci_all = pred_future.conf_int()

yhat_future = yhat_all.iloc[TEST_SIZE:]
ci_future = ci_all.iloc[TEST_SIZE:]

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(11, 5))
plt.plot(y.index.to_timestamp(), y.values, label="Observed")
plt.plot(y_test.index.to_timestamp(), yhat_test.values, label="Test prediction")
plt.plot(yhat_future.index.to_timestamp(), yhat_future.values, label=f"Forecast (+{H})")

plt.fill_between(ci_test.index.to_timestamp(),
                 ci_test.iloc[:, 0].values, ci_test.iloc[:, 1].values, alpha=0.2)
plt.fill_between(ci_future.index.to_timestamp(),
                 ci_future.iloc[:, 0].values, ci_future.iloc[:, 1].values, alpha=0.2)

plt.axvline(y_test.index[0].to_timestamp(), linestyle="--")
plt.title(f"SARIMAX{order} x Seasonal{seasonal_order} Forecast")
plt.xlabel("Time")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()

# Diagnostics
fit.plot_diagnostics(figsize=(11, 6))
plt.tight_layout()
plt.show()
