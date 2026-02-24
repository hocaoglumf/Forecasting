from statsmodels.tsa.holtwinters import Holt

data = [120, 128, 135, 145, 152, 160]

model = Holt(data)
fit = model.fit(smoothing_level=0.4,
                smoothing_trend=0.3)
# alpha = 0.4
# beta = 0.3
forecast = fit.forecast(3)

print("Forecast:", forecast)