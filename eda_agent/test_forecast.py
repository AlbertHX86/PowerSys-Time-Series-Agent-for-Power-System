"""
Test script to verify forecast functionality with actual data
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Load data
print("Loading data...")
df = pd.read_excel('data/Solar station site 1 (Nominal capacity-50MW)(1).xlsx')
print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Prepare features and target
target_col = 'Power (MW)'
time_col = 'Time(year-month-day h:m:s)'
feature_cols = [col for col in df.columns if col not in [target_col, time_col]]

print(f"\nFeature columns: {feature_cols}")
print(f"Target column: {target_col}")

X = df[feature_cols].values
y = df[target_col].values

# Train/test split (80/20)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# Train Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluate on test set
rf_pred = rf_model.predict(X_test)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print(f"Random Forest - R²: {rf_r2:.4f}, MAE: {rf_mae:.4f}, RMSE: {rf_rmse:.4f}")

# Now test forecasting using last 24 hours
print("\n" + "="*60)
print("TESTING FORECAST GENERATION")
print("="*60)

# Get last 24 hours of data
X_last_24 = df[feature_cols].tail(24).values
y_last_24_actual = df[target_col].tail(24).values

print(f"\nLast 24 hours features shape: {X_last_24.shape}")
print(f"\nActual power values (last 24 hours):")
for i in range(24):
    time_val = df[time_col].iloc[-24+i]
    print(f"  Hour {i+1:2d} ({time_val}): Actual={y_last_24_actual[i]:.2f} MW")

# Generate forecast using the model
print(f"\n" + "-"*60)
print("FORECAST: Using last 24 hours features to predict next day")
print("-"*60)

forecast_values = []
for i in range(24):
    X_input = X_last_24[i:i+1]
    pred = rf_model.predict(X_input)[0]
    pred = max(0, pred)  # Ensure non-negative
    forecast_values.append(pred)

print(f"\nForecast results:")
for i in range(24):
    # Calculate what the time would be for tomorrow
    hour = (18 + i) % 24  # Starting from 18:00 (last data timestamp)
    print(f"  Hour {i+1:2d} (+{i+1}h): Forecast={forecast_values[i]:.2f} MW  (template from yesterday hour {hour}:00)")

# Calculate statistics
forecast_mean = np.mean(forecast_values)
forecast_std = np.std(forecast_values)
forecast_min = np.min(forecast_values)
forecast_max = np.max(forecast_values)

print(f"\n" + "="*60)
print("FORECAST STATISTICS")
print("="*60)
print(f"Mean:   {forecast_mean:.2f} MW")
print(f"Std:    {forecast_std:.2f} MW")
print(f"Min:    {forecast_min:.2f} MW")
print(f"Max:    {forecast_max:.2f} MW")
print(f"Range:  {forecast_min:.2f} - {forecast_max:.2f} MW")

# Check if forecast makes sense
if forecast_max > 0:
    print(f"\n✅ SUCCESS: Forecast has variation (not all zeros)")
    print(f"   Peak forecast: {forecast_max:.2f} MW")
else:
    print(f"\n❌ PROBLEM: All forecast values are zero!")
    print(f"   This means the input features are likely all zeros (nighttime data)")
    print(f"   Solution: Use features from a full day cycle (daytime hours)")

# Show feature values for verification
print(f"\n" + "="*60)
print("FEATURE VALUES USED FOR FORECAST (sample)")
print("="*60)
print(f"First hour features (nighttime):")
for j, fcol in enumerate(feature_cols):
    print(f"  {fcol}: {X_last_24[0, j]:.2f}")

if len(X_last_24) > 12:
    print(f"\nMidday hour features (around noon):")
    for j, fcol in enumerate(feature_cols):
        print(f"  {fcol}: {X_last_24[12, j]:.2f}")
