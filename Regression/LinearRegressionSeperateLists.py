from typing import List, Tuple

def linear_regression(X: List[float], Y: List[float]) -> Tuple[float, float]:
    """
    Computes simple linear regression (Y = a + bX)

    Parameters:
        X : List of independent variable values
        Y : List of dependent variable values

    Returns:
        a : Intercept
        b : Slope
    """

    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length.")

    n = len(X)

    if n == 0:
        raise ValueError("Input lists must not be empty.")

    sum_x = sum(X)
    sum_y = sum(Y)
    sum_xy = sum(x * y for x, y in zip(X, Y))
    sum_x2 = sum(x ** 2 for x in X)

    # Slope (b)
    b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

    # Intercept (a)
    a = (sum_y - b * sum_x) / n

    return a, b


def predict(X: List[float], a: float, b: float) -> List[float]:
    """
    Predict Y values using regression model
    """

    return [a + b * x for x in X]


# =============================
# Example Usage
# =============================

Y = [264, 116, 165, 101, 209]      # Independent variable
X = [2.5, 1.3, 1.4,1.0,2.0]      # Dependent variable

a, b = linear_regression(X, Y)

print(f"Intercept (a): {a:.4f}")
print(f"Slope (b): {b:.4f}")

Y_pred = predict(X, a, b)

print("Predicted Y:", Y_pred)
