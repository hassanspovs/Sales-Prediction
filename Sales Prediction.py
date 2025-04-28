import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

np.random.seed(42)

sales_data = {
    'marketing_spend': np.random.randint(1000, 10000, 300),
    'store_visits': np.random.randint(100, 1000, 300),
    'product_price': np.random.uniform(5, 50, 300),
    'social_media_spend': np.random.randint(500, 7000, 300),
    'discount_offered': np.random.uniform(0, 30, 300),
    'competitor_price': np.random.uniform(5, 50, 300),
    'holiday_season': np.random.choice([0, 1], 300),
    'sales': np.random.randint(5000, 50000, 300)
}

df = pd.DataFrame(sales_data)

X = df.drop('sales', axis=1)
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")

importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10,6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
