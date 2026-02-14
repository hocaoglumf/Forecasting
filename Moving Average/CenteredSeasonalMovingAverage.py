def simple_moving_average(data, window):
    """
    Computes simple moving average.
    Skips windows that contain None.

    Returns list with None at invalid positions.
    """
    n = len(data)
    ma = [None] * n

    half = window // 2

    for i in range(n):

        start = i - half
        end = i + half + 1

        # Adjustment for even window
        if window % 2 == 0:
            start += 1

        # Boundary check
        if start < 0 or end > n:
            continue

        window_vals = data[start:end]

        # Skip if any None in window
        if any(v is None for v in window_vals):
            continue

        s = 0.0
        for v in window_vals:
            s += v

        ma[i] = s / window

    return ma


def centered_seasonal_moving_average(data, s):
    """
    Centered Seasonal Moving Average (Classical Decomposition)

    data : list of floats
    s    : seasonal period (12=monthly, 4=quarterly, etc.)

    Returns trend list (with None at boundaries)
    """

    # Step 1: s-period moving average
    ma_s = simple_moving_average(data, s)

    # Step 2: If s is even → 2-MA centering
    if s % 2 == 0:
        trend = simple_moving_average(ma_s, 2)
        return trend
    else:
        return ma_s


# --------------------------------
# Example usage
# --------------------------------
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

s = 12  # monthly seasonality

trend = centered_seasonal_moving_average(y, s)

# Print result
print(" t    y        Trend")
print("---------------------------")

for i, (obs, tr) in enumerate(zip(y, trend), start=1):
    if tr is None:
        print(f"{i:2d}  {obs:8.2f}   None")
    else:
        print(f"{i:2d}  {obs:8.2f}   {tr:8.2f}")
