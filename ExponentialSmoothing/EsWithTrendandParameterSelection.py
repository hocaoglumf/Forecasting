import matplotlib.pyplot as plt

# ----------------------------
# Holt's Linear Trend (Double Exponential Smoothing) - Pure Python
# ----------------------------
def holt_linear_smoothing(data, alpha, beta):
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")
    if not (0 < beta < 1):
        raise ValueError("beta must be between 0 and 1")
    if len(data) == 0:
        return [], [], []

    n = len(data)
    level = [0.0] * n
    trend = [0.0] * n
    fitted = [None] * n  # fitted[t] forecasts y[t]; fitted[0] undefined

    # Initialization
    level[0] = float(data[0])
    trend[0] = float(data[1] - data[0]) if n > 1 else 0.0

    # Recursion
    for t in range(1, n):
        fitted[t] = level[t - 1] + trend[t - 1]  # 1-step forecast for y[t]
        level[t] = alpha * float(data[t]) + (1 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]

    return level, trend, fitted


def holt_forecast(data, alpha, beta, h):
    level, trend, _ = holt_linear_smoothing(data, alpha, beta)
    if not level:
        return [0.0] * h
    L = level[-1]
    B = trend[-1]
    return [L + (m + 1) * B for m in range(h)]


def holt_sse(data, alpha, beta):
    _, _, fitted = holt_linear_smoothing(data, alpha, beta)
    sse = 0.0
    count = 0
    for t in range(1, len(data)):
        if fitted[t] is None:
            continue
        err = float(data[t]) - float(fitted[t])
        sse += err * err
        count += 1
    return sse, count


def grid_search_alpha_beta(data, alpha_step=0.02, beta_step=0.02):
    best = {"alpha": None, "beta": None, "mse": float("inf")}
    a = alpha_step
    while a < 1.0:
        b = beta_step
        while b < 1.0:
            sse, n = holt_sse(data, a, b)
            if n > 0:
                mse = sse / n
                if mse < best["mse"]:
                    best.update({"alpha": a, "beta": b, "mse": mse})
            b = round(b + beta_step, 10)
        a = round(a + alpha_step, 10)

    if best["alpha"] is None:
        raise RuntimeError("Grid search failed.")
    return best


# ----------------------------
# Example data (replace with yours)
# ----------------------------
'''
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
'''
y=[1325, 1353, 1305, 1275, 1210]
# ----------------------------
# Parameter selection
# ----------------------------
best = grid_search_alpha_beta(y, alpha_step=0.02, beta_step=0.02)
alpha_best, beta_best, mse_best = best["alpha"], best["beta"], best["mse"]

print("Best parameters (grid search):")
print("  alpha =", alpha_best)
print("  beta  =", beta_best)
print("  MSE   =", mse_best)

# ----------------------------
# Fit + Forecast
# ----------------------------
level, trend, fitted = holt_linear_smoothing(y, alpha_best, beta_best)

H = 12
future = holt_forecast(y, alpha_best, beta_best, H)

t_obs = list(range(1, len(y) + 1))
t_future = list(range(len(y) + 1, len(y) + H + 1))

print("Future:", future)

fitted_x = [t_obs[i] for i in range(len(fitted)) if fitted[i] is not None]
fitted_y = [fitted[i] for i in range(len(fitted)) if fitted[i] is not None]

# ----------------------------
# Plot: Observed + Fitted + Forecast
# ----------------------------
plt.figure(figsize=(11, 5))
plt.plot(t_obs, y, label="Observed")
plt.plot(fitted_x, fitted_y, label="Fitted (Holt trend)")
plt.plot(t_future, future, label=f"Forecast (+{H})")

plt.axvline(x=len(y), linestyle="--")
plt.title(f"Holt Trend Exponential Smoothing (alpha={alpha_best:.2f}, beta={beta_best:.2f}, MSE={mse_best:.2f})")
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()
