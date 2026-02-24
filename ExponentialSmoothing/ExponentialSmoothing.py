# pip install pandas numpy matplotlib statsmodels

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -----------------------------
# 1) Veriyi girin
# -----------------------------
y = [
    280.05, 305.18, 324.14, 355.64, 394.53, 438.96, 501.13, 572.49, 635.81,
    726.68, 784.94, 727.34, 713.08, 747.34, 771.44, 894.97, 894.46, 968.30,
    1012.44, 1046.07, 1132.78, 1252.69, 1187.70, 1205.45, 1241.29, 1319.69,
    1426.17, 1856.83, 2023.56, 2129.56, 2239.85, 2389.03, 2660.00, 2770.00,
    2863.00, 3228.00, 3557.00, 3894.00, 4171.00, 4580.00, 4644.00, 4621.00,
    5142.00, 6440.00, 6448.00, 5960.00, 6155.00, 6507.00, 6001.00, 5927.00,
    6731.00, 7614.00, 7103.00, 6746.00, 7328.00, 8212.00, 7993.00, 8300.00,
    8731.00, 8702.00, 6677.00
]

# Zaman endeksi (aylık varsayımıyla). İsterseniz değiştirin.
idx = pd.period_range(start="2021-01", periods=len(y), freq="M")
ts = pd.Series(y, index=idx).astype(float)

# -----------------------------
# 2) Holt-Winters (Seasonal Exponential Smoothing) modeli
# -----------------------------
SEASONAL_PERIODS = 12     # aylık mevsimsellik varsayımı
H = 12                    # kaç dönem ileri tahmin

# Mevsimsellik türü:
# seasonal="add" : genlik sabit (eklemeli)
# seasonal="mul" : seviye arttıkça genlik artar (çarpanlı) -> sizin seri gibi büyüyen serilerde sık iyi çalışır.
model = ExponentialSmoothing(
    ts,
    trend="add",           # "add" veya "mul"
    seasonal="mul",        # "add" veya "mul"
    seasonal_periods=SEASONAL_PERIODS,
    initialization_method="estimated"
)

fit = model.fit(optimized=True, use_brute=True)

# -----------------------------
# 3) Uyum (fitted) ve Tahmin
# -----------------------------
fitted = fit.fittedvalues
forecast = fit.forecast(H)

# -----------------------------
# 4) Çıktılar
# -----------------------------
print("=== Model Özeti ===")
print(fit.summary())

print("\n=== Parametreler ===")
print(f"alpha (level)   : {fit.params.get('smoothing_level')}")
print(f"beta  (trend)   : {fit.params.get('smoothing_trend')}")
print(f"gamma (seasonal): {fit.params.get('smoothing_seasonal')}")

# -----------------------------
# 5) Grafik
# -----------------------------
plt.figure(figsize=(11, 5))
plt.plot(ts.to_timestamp(), ts.values, label="Observed")
plt.plot(fitted.to_timestamp(), fitted.values, label="Fitted")
plt.plot(forecast.to_timestamp(), forecast.values, label=f"Forecast ({H} steps)")
plt.title("Holt-Winters Seasonal Exponential Smoothing")
plt.xlabel("Time")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()
