import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Your data
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

df = pd.DataFrame({"y": y})

H = 12
window = 5  # moving average window size
df["SMA"] = df["y"].rolling(window=window).mean()

# Plot
plt.figure(figsize=(11,5))
plt.plot(df["y"], label="Observed")
plt.plot(df["SMA"], label=f"{window}-period SMA")
plt.legend()
plt.title("Simple Moving Average")
plt.show()

#import numpy as np


def moving_average_forecast(series, window, steps):
    series = list(series)
    forecasts = []

    for _ in range(steps):
        if len(series) < window:
            raise ValueError("Series shorter than window size.")
        next_val = np.mean(series[-window:])
        forecasts.append(next_val)
        series.append(next_val)  # recursive

    return forecasts



forecast = moving_average_forecast(y, window, H)

print("Forecast values:")
print(forecast)




#import numpy as np

y_np = np.array(y)

sma = np.convolve(y_np, np.ones(window)/window, mode='valid')

print("SMA values:")
print(sma)


def weighted_moving_average(series, weights):
    weights = np.array(weights)
    window = len(weights)
    wma = []

    for i in range(window - 1, len(series)):
        window_data = series[i - window + 1:i + 1]
        value = np.dot(window_data, weights) / weights.sum()
        wma.append(value)

    return wma


weights = [1, 2, 3, 4, 5]  # increasing weights
wma_values = weighted_moving_average(y, weights)

print("Weighted MA:")
print(wma_values)


def seasonal_ma_forecast(series, s=12, k=3, h=12):
    '''

    :param series:
    :param s: Seasonal Period (Season Length)
    :param k: Number of Past Seasons Used (Averaging Window)
    :param h: Forecast Horizon (How Far Ahead)
    :return:
    '''
    series = list(series)
    out = []

    for i in range(h):
        season_vals = []
        for j in range(1, k+1):
            idx = -j*s + i % s
            season_vals.append(series[idx])

        out.append(np.mean(season_vals))

    return out


forecast = seasonal_ma_forecast(y, s=12, k=3, h=12)
print("Seasonel Moving Average")
print(forecast)
