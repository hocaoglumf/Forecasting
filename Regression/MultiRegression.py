import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# ----------------------------
# 1) Data
# ----------------------------
y_list = [
    280.05, 305.18, 324.14, 355.64, 394.53, 438.96, 501.13, 572.49, 635.81,
    726.68, 784.94, 727.34, 713.08, 747.34, 771.44, 894.97, 894.46, 968.30,
    1012.44, 1046.07, 1132.78, 1252.69, 1187.70, 1205.45, 1241.29, 1319.69,
    1426.17, 1856.83, 2023.56, 2129.56, 2239.85, 2389.03, 2660.00, 2770.00,
    2863.00, 3228.00, 3557.00, 3894.00, 4171.00, 4580.00, 4644.00, 4621.00,
    5142.00, 6440.00, 6448.00, 5960.00, 6155.00, 6507.00, 6001.00, 5927.00,
    6731.00, 7614.00, 7103.00, 6746.00, 7328.00, 8212.00, 7993.00, 8300.00,
    8731.00, 8702.00, 6677.00
]

y = pd.Series(y_list, dtype=float)

# Varsayım: aylık veri → seasonal_periods = 12
SEASONAL_PERIODS = 12
H = 12
TEST_SIZE = 12

# ----------------------------
# 2) Feature engineering
# ----------------------------
df = pd.DataFrame({"y": y})
df["t"] = np.arange(1, len(df) + 1)

df["lag1"] = df["y"].shift(1)
df[f"lag{SEASONAL_PERIODS}"] = df["y"].shift(SEASONAL_PERIODS)

# "month" bilgisini PeriodIndex olmadan da üretebiliriz:
# eğer veri aylıksa ve t=1 Ocak kabul edilecekse:
df["month"] = ((df["t"] - 1) % SEASONAL_PERIODS) + 1  # 1..12

df_model = df.dropna().copy()

# ----------------------------
# 3) Train/test split (time order)
# ----------------------------
train = df_model.iloc[:-TEST_SIZE].copy()
test  = df_model.iloc[-TEST_SIZE:].copy()

# ----------------------------
# 4) Fit: Multiple regression with seasonality as categorical
# ----------------------------
formula = f"y ~ t + lag1 + lag{SEASONAL_PERIODS} + C(month)"
model = smf.ols(formula, data=train).fit()

print(model.summary())

# Test prediction
test_pred = model.predict(test)

# ----------------------------
# 5) Recursive forecast H steps
# ----------------------------
history = df["y"].copy()  # original series (no NaNs)
future_preds = []
future_rows = []

for step in range(1, H + 1):
    t_val = len(history) + 1

    lag1 = history.iloc[-1]
    lagS = history.iloc[-SEASONAL_PERIODS]  # lag12 (monthly)

    month = ((t_val - 1) % SEASONAL_PERIODS) + 1

    row = {
        "t": t_val,
        "lag1": float(lag1),
        f"lag{SEASONAL_PERIODS}": float(lagS),
        "month": int(month)
    }
    y_hat = float(model.predict(pd.DataFrame([row]))[0])

    future_rows.append(row)
    future_preds.append(y_hat)
    history = pd.concat([history, pd.Series([y_hat])], ignore_index=True)

forecast = pd.Series(future_preds, name="forecast")

# ----------------------------
# 6) Plot
# ----------------------------
plt.figure(figsize=(11, 5))
plt.plot(df["t"], df["y"], label="Observed")
plt.plot(test["t"], test_pred, label="Test prediction")
plt.plot(np.arange(df["t"].iloc[-1] + 1, df["t"].iloc[-1] + H + 1), forecast, label=f"Forecast (+{H})")
plt.axvline(test["t"].iloc[0] - 0.5, linestyle="--")
plt.xlabel("t")
plt.ylabel("y")
plt.title("Multiple Regression Forecasting (OLS: trend + lags + seasonality)")
plt.legend()
plt.tight_layout()
plt.show()
