import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ----------------------------
# Data (your series)
# ----------------------------
y = np.array([
    280.05, 305.18, 324.14, 355.64, 394.53, 438.96, 501.13, 572.49, 635.81,
    726.68, 784.94, 727.34, 713.08, 747.34, 771.44, 894.97, 894.46, 968.30,
    1012.44, 1046.07, 1132.78, 1252.69, 1187.70, 1205.45, 1241.29, 1319.69,
    1426.17, 1856.83, 2023.56, 2129.56, 2239.85, 2389.03, 2660.00, 2770.00,
    2863.00, 3228.00, 3557.00, 3894.00, 4171.00, 4580.00, 4644.00, 4621.00,
    5142.00, 6440.00, 6448.00, 5960.00, 6155.00, 6507.00, 6001.00, 5927.00,
    6731.00, 7614.00, 7103.00, 6746.00, 7328.00, 8212.00, 7993.00, 8300.00,
    8731.00, 8702.00, 6677.00
], dtype=float)

#y = np.array([
#    264, 116, 165, 101, 209
#] , dtype=float)


n = len(y)
t = np.arange(1, n + 1)  # 1..n

# ----------------------------
# Train/test split (no shuffle)
# ----------------------------
test_size = 12
t_train, t_test = t[:-test_size], t[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

# Add constant (intercept)
X_train = sm.add_constant(t_train)
X_test  = sm.add_constant(t_test)

# ----------------------------
# Fit OLS
# ----------------------------
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

# Test prediction
y_pred_test = model.predict(X_test)

# ----------------------------
# Forecast future H steps
# ----------------------------
H = 12
t_future = np.arange(n + 1, n + H + 1)
X_future = sm.add_constant(t_future)
y_forecast = model.predict(X_future)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(11, 5))
plt.plot(t, y, label="Observed")
plt.plot(t_test, y_pred_test, label="Test prediction")
plt.plot(t_future, y_forecast, label=f"Forecast (+{H})")
plt.axvline(n - test_size + 0.5, linestyle="--")
plt.xlabel("t")
plt.ylabel("y")
plt.title("Linear Regression Forecasting (statsmodels OLS)")
plt.legend()
plt.tight_layout()
plt.show()
