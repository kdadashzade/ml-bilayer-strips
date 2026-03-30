import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import make_pipeline

# 1. Load data, remove outliers
csv_file = 'Bimetal_Dataset_200Runs_1.csv'
df = pd.read_csv(csv_file)

df['Slenderness'] = df['Length_mm'] / (df['TPU_Thick_mm'] + df['PLA_Thick_mm'])
df_clean = df[df['Slenderness'] < 400].copy()
print(f"Removed {len(df) - len(df_clean)} extreme outlier(s). \n")

X = df_clean[['Length_mm', 'TPU_Thick_mm', 'PLA_Thick_mm']]
y = df_clean['Deflection_mm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

# 2. TRAIN PURE LINEAR MODEL
# ---------------------------------------------------------
# No PolynomialFeatures, just a straightforward Linear Regression
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# ---------------------------------------------------------
# 3. CALCULATE & PRINT METRICS
# ---------------------------------------------------------
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

mape_train = mean_absolute_percentage_error(y_train, y_train_pred) * 100
mape_test = mean_absolute_percentage_error(y_test, y_test_pred) * 100

print("--- Model Performance ---")
print(f"Training Data -> RMSE: {rmse_train:.4f} | R-squared: {r2_train:.4f} | Error: {mape_train:.2f}%")
print(f"Testing Data  -> RMSE: {rmse_test:.4f} | R-squared: {r2_test:.4f} | Error: {mape_test:.2f}%")

# Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("\n--- Cross-Validation Results ---")
print(f"R-squared scores for 5 folds: {np.round(cv_scores, 4)}")
print(f"Average CV R-squared: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Feature Importance
linear_model = model.named_steps['linearregression']
coefficients = linear_model.coef_

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient_Magnitude': np.abs(coefficients),
    'Actual_Weight': coefficients
}).sort_values(by='Coefficient_Magnitude', ascending=False)

print("\n--- Feature Importance (Linear Weights) ---")
print(importance_df.to_string(index=False))

# ---------------------------------------------------------
# 4. VISUALIZATIONS
# ---------------------------------------------------------
# Plot 1: Actual vs. Predicted
plt.figure(1, figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.7, label='Test Data')
plt.scatter(y_train, y_train_pred, color='green', alpha=0.3, label='Train Data')

min_val = min(y.min(), min(y_test_pred), min(y_train_pred))
max_val = max(y.max(), max(y_test_pred), max(y_train_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Fit')

plt.title('Pure Linear Regression: Actual vs. Predicted')
plt.xlabel('Actual Deflection (mm)')
plt.ylabel('Predicted Deflection (mm)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Plot 2: Residual Plot
plt.figure(2, figsize=(8, 6))
plt.scatter(y_train_pred, y_train - y_train_pred, color='green', alpha=0.4, label='Train Residuals')
plt.scatter(y_test_pred, y_test - y_test_pred, color='blue', alpha=0.6, label='Test Residuals')

plt.hlines(y=0, xmin=min(y_test_pred.min(), y_train_pred.min()),
           xmax=max(y_test_pred.max(), y_train_pred.max()), color='red', linestyle='--')

plt.title('Residual Plot: Linear Model Error Distribution')
plt.xlabel('Predicted Deflection (mm)')
plt.ylabel('Residual Error (Actual - Predicted) (mm)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.show()