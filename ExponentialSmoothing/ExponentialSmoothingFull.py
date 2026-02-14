"""
exponential_smoothing_all.py

Merged Exponential Smoothing methods in ONE file:

1) Simple Exponential Smoothing (SES)
2) Holt's Linear Trend (Double Exponential Smoothing)
3) Holt-Winters Seasonal (Triple Exponential Smoothing) via statsmodels
4) SES best alpha selection (grid search)

Author: (your name)
"""

from __future__ import annotations
from typing import List, Tuple, Optional


# ============================================================
# 1) Simple Exponential Smoothing (SES) - Pure Python
# ============================================================

def ses_fit(data: List[float], alpha: float) -> List[float]:
    """
    Simple Exponential Smoothing (one-step-ahead forecasts).
    Forecast[t] is the forecast for period t (based on data up to t-1).

    Initialization: forecast[0] = data[0]

    Formula:
        F_t = alpha * y_{t-1} + (1-alpha) * F_{t-1}   for t>=1
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1")

    if len(data) == 0:
        return []

    f = [0.0] * len(data)
    f[0] = float(data[0])

    for t in range(1, len(data)):
        f[t] = alpha * float(data[t - 1]) + (1.0 - alpha) * f[t - 1]

    return f


def ses_forecast(data: List[float], alpha: float, h: int) -> List[float]:
    """
    SES h-step-ahead forecast.
    In SES, all future forecasts equal the last smoothed level.
    """
    if h < 0:
        raise ValueError("h must be non-negative")
    if len(data) == 0:
        return [0.0] * h

    f = ses_fit(data, alpha)
    last_level = f[-1]
    return [last_level] * h


def _mse(y_true: List[float], y_pred: List[float]) -> float:
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true and y_pred must have same non-zero length")
    s = 0.0
    for a, b in zip(y_true, y_pred):
        d = a - b
        s += d * d
    return s / len(y_true)


def ses_best_alpha(data: List[float], step: float = 0.01) -> Tuple[float, float]:
    """
    Grid-search alpha in (0,1) to minimize one-step-ahead MSE on in-sample forecasts.
    Uses forecasts for t>=1 compared against actual y[t].
    """
    if not (0.0 < step < 1.0):
        raise ValueError("step must be between 0 and 1")

    if len(data) < 2:
        raise ValueError("Need at least 2 observations to select alpha")

    best_a = None
    best_err = float("inf")

    # alpha candidates: step, 2*step, ..., <1
    kmax = int(1.0 / step)
    for k in range(1, kmax):
        a = k * step
        f = ses_fit(data, a)

        # one-step forecasts for t>=1 are f[t]
        y_true = [float(v) for v in data[1:]]
        y_pred = [float(v) for v in f[1:]]
        err = _mse(y_true, y_pred)

        if err < best_err:
            best_err = err
            best_a = a

    # best_a cannot be None here if step is valid
    return float(best_a), float(best_err)


# ============================================================
# 2) Holt's Linear Trend - Pure Python (Double Exponential)
# ============================================================

def holt_fit(data: List[float], alpha: float, beta: float) -> Tuple[List[float], List[float], List[float]]:
    """
    Holt's Linear Trend method (double exponential smoothing).

    Level:  L_t = alpha*y_t + (1-alpha)*(L_{t-1} + B_{t-1})
    Trend:  B_t = beta*(L_t - L_{t-1}) + (1-beta)*B_{t-1}
    Forecast (one-step): F_{t+1} = L_t + B_t

    Returns:
        level list L
        trend list B
        one-step forecasts list F (aligned with t index: F[t] forecasts y[t], for t>=1)
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1")
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be between 0 and 1")
    if len(data) == 0:
        return [], [], []

    y = [float(v) for v in data]
    n = len(y)

    L = [0.0] * n
    B = [0.0] * n
    F = [None] * n  # F[t] forecasts y[t], so F[0] undefined

    # Initialization (simple):
    L[0] = y[0]
    B[0] = (y[1] - y[0]) if n >= 2 else 0.0

    for t in range(1, n):
        # one-step forecast for y[t] based on t-1:
        F[t] = L[t-1] + B[t-1]

        L[t] = alpha * y[t] + (1.0 - alpha) * (L[t-1] + B[t-1])
        B[t] = beta * (L[t] - L[t-1]) + (1.0 - beta) * B[t-1]

    return L, B, F


def holt_forecast(data: List[float], alpha: float, beta: float, h: int) -> List[float]:
    """
    Holt h-step-ahead forecast: y_{n+m} = L_{n-1} + m*B_{n-1},  m=1..h
    """
    if h < 0:
        raise ValueError("h must be non-negative")
    if len(data) == 0:
        return [0.0] * h

    L, B, _ = holt_fit(data, alpha, beta)
    last_L = L[-1]
    last_B = B[-1]
    return [last_L + (m * last_B) for m in range(1, h + 1)]


# ============================================================
# 3) Holt-Winters Seasonal (Triple Exponential) - statsmodels
# ============================================================

def holt_winters_fit_forecast(
    data: List[float],
    seasonal_periods: int,
    h: int,
    trend: str = "add",
    seasonal: str = "mul",
    damped_trend: bool = False
) -> Tuple[object, List[float], List[float]]:
    """
    Holt-Winters seasonal exponential smoothing using statsmodels.

    Parameters:
        seasonal_periods: e.g. 12 for monthly, 4 quarterly
        h: forecast horizon
        trend: "add" or "mul" or None
        seasonal: "add" or "mul" or None
        damped_trend: True/False

    Returns:
        fitted_model, fitted_values(list), forecast_values(list)
    """
    if h < 0:
        raise ValueError("h must be non-negative")
    if seasonal_periods <= 1:
        raise ValueError("seasonal_periods must be >= 2 for seasonal models")
    if len(data) < 2 * seasonal_periods:
        raise ValueError("For Holt-Winters, it's recommended to have at least 2 full seasons of data.")

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError as e:
        raise ImportError("statsmodels is required for Holt-Winters seasonal model. Install: pip install statsmodels") from e

    import pandas as pd

    ts = pd.Series([float(v) for v in data])

    model = ExponentialSmoothing(
        ts,
        trend=trend,
        damped_trend=damped_trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated"
    )

    fit = model.fit(optimized=True, use_brute=True)

    fitted_vals = [float(v) for v in fit.fittedvalues.tolist()]
    forecast_vals = [float(v) for v in fit.forecast(h).tolist()]

    return fit, fitted_vals, forecast_vals


# ============================================================
# 4) Demo Runner (optional)
# ============================================================

def _demo():
    data = [
        280.05, 305.18, 324.14, 355.64, 394.53, 438.96, 501.13, 572.49, 635.81,
        726.68, 784.94, 727.34, 713.08, 747.34, 771.44, 894.97, 894.46, 968.30,
        1012.44, 1046.07, 1132.78, 1252.69, 1187.70, 1205.45, 1241.29, 1319.69,
        1426.17, 1856.83, 2023.56, 2129.56, 2239.85, 2389.03, 2660.00, 2770.00,
        2863.00, 3228.00, 3557.00, 3894.00, 4171.00, 4580.00, 4644.00, 4621.00,
        5142.00, 6440.00, 6448.00, 5960.00, 6155.00, 6507.00, 6001.00, 5927.00,
        6731.00, 7614.00, 7103.00, 6746.00, 7328.00, 8212.00, 7993.00, 8300.00,
        8731.00, 8702.00, 6677.00
    ]

    print("=== SES ===")
    alpha = 0.3
    ses_vals = ses_fit(data, alpha)
    print("Last SES level:", ses_vals[-1])
    print("SES 12-step forecast:", ses_forecast(data, alpha, 12))

    print("\n=== SES best alpha (grid) ===")
    best_a, best_mse = ses_best_alpha(data, step=0.01)
    print("best alpha:", best_a, "MSE:", best_mse)

    print("\n=== Holt (trend) ===")
    alpha_h, beta_h = 0.3, 0.2
    L, B, F = holt_fit(data, alpha_h, beta_h)
    print("Last level:", L[-1], "Last trend:", B[-1])
    print("Holt 12-step forecast:", holt_forecast(data, alpha_h, beta_h, 12))

    print("\n=== Holt-Winters (seasonal) via statsmodels ===")
    try:
        fit, fitted_vals, fc = holt_winters_fit_forecast(
            data=data,
            seasonal_periods=12,
            h=12,
            trend="add",
            seasonal="mul",
            damped_trend=False
        )
        print("HW 12-step forecast:", fc)
        print("HW params:", {k: fit.params[k] for k in fit.params if 'smoothing' in k})
    except Exception as e:
        print("Holt-Winters demo skipped:", e)

    # Optional plot if matplotlib exists
    try:
        import matplotlib.pyplot as plt
        x = list(range(1, len(data) + 1))
        plt.figure(figsize=(11, 5))
        plt.plot(x, data, label="Observed")
        plt.plot(x, ses_vals, label=f"SES(alpha={alpha})")
        plt.title("Exponential Smoothing Methods (Observed vs SES)")
        plt.xlabel("t")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    _demo()
