from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Create lagged dataset
X = []
y = []

series = [120,130,150,170,190,210,230,220,200]

for i in range(3, len(series)):
    X.append(series[i-3:i])
    y.append(series[i])

print("Original Data", series)
print("Lagged Data")
for i in range(len(X)):
    print("X: ", X[i], "  Y:",y[i])

model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42)
model.fit(X, y)

prediction = model.predict([X[-1]])
print("Prediction: ",prediction)