import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------
# 1) Create lagged dataset
# -----------------------------
def make_lags(series, p):
    X, y = [], []
    for t in range(p, len(series)):
        X.append(series[t-p:t])      # last p values
        y.append(series[t])
    return np.array(X), np.array(y)

#series = np.array([280.05, 305.18, 324.14, 355.64, 394.53, 438.96, 501.13, 572.49])

series=np.array([
            280.05, 305.18, 324.14, 355.64, 394.53, 438.96, 501.13, 572.49, 635.81,
            726.68, 784.94, 727.34, 713.08, 747.34, 771.44, 894.97, 894.46, 968.30,
            1012.44, 1046.07, 1132.78, 1252.69, 1187.70, 1205.45, 1241.29, 1319.69,
            1426.17, 1856.83, 2023.56, 2129.56, 2239.85, 2389.03, 2660.00, 2770.00,
            2863.00, 3228.00, 3557.00, 3894.00, 4171.00, 4580.00, 4644.00, 4621.00,
            5142.00, 6440.00, 6448.00, 5960.00, 6155.00, 6507.00, 6001.00, 5927.00,
            6731.00, 7614.00, 7103.00, 6746.00, 7328.00, 8212.00, 7993.00, 8300.00,
            8731.00, 8702.00, 6677.00
        ])

p = 4
X, y = make_lags(series, p)

print("Lags: X")
for i in range(len(X)):
    print(y[i],"--->",X[i])

# -----------------------------
# 2) Train/test split (no shuffle)
# -----------------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# 3) Scaling (simple standardization)
# -----------------------------
mu = X_train.mean()
sigma = X_train.std() + 1e-9

X_train = (X_train - mu) / sigma
X_test  = (X_test - mu) / sigma

# -----------------------------
# 4) Build ANN (MLP)
# -----------------------------
model = keras.Sequential([
    layers.Input(shape=(p,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# -----------------------------
# 5) Train
# -----------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=500,
    batch_size=8,
    callbacks=[early_stop],
    verbose=0
)

# -----------------------------
# 6) Predict
# -----------------------------
pred = model.predict(X_test).flatten()
print("Test predictions:", pred)
