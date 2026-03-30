import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate


#################

df = pd.read_csv('Bimetal_Dataset_200Runs.csv')
X = df[['Length_mm', 'TPU_Thick_mm', 'PLA_Thick_mm']]
y = df[['Deflection_mm']]

df.rename(columns={"Length_mm": "len", "TPU_Thick_mm": "TPU", "PLA_Thick_mm" : "PLA", "Deflection_mm": "U"})

###############

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

print(f"Training Features Shape: {X_train.shape}")
print(f"Testing Features Shape:  {X_test.shape}")

###############
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#################
poly = PolynomialFeatures(degree=6, include_bias=False)

X_train_poly = poly.fit_transform(X_train_scaled)

X_test_poly = poly.transform(X_test_scaled)

print(f"Original Feature Count: {X_train_scaled.shape[1]}")
print(f"Polynomial Feature Count: {X_train_poly.shape[1]}")
print("Feature Names:", poly.get_feature_names_out(['L', 'T', 'P']))

model = LinearRegression()
model.fit(X_train_poly, y_train)

#############
intercept_val = np.ravel(model.intercept_)[0]
print(f"\nModel Intercept (Baseline Deflection): {intercept_val:.4f}")

y_train_pred = model.predict(X_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)

y_test_pred = model.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n--- Single Split Performance ---")
print(f"Training RMSE: {train_rmse:.4f} mm")
print(f"Testing RMSE:  {test_rmse:.4f} mm")
print(f"Testing R-squared: {test_r2:.4f}")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=6, include_bias=False)),
    ('model', LinearRegression())
])

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

cv_results = cross_validate(
    pipeline, X, y,
    cv=kfold,
    scoring='neg_mean_squared_error',
    return_train_score=True
)

# Extract and convert the negative MSE to RMSE
train_rmse_scores = np.sqrt(-cv_results['train_score'])
test_rmse_scores = np.sqrt(-cv_results['test_score'])

print(f"\n--- 10-Fold Cross-Validation Performance")
print(f"Train RMSE across 10 folds: {np.round(train_rmse_scores, 4)}")
print(f"Test RMSE across 10 folds:  {np.round(test_rmse_scores, 4)}\n")

print(f"Average CV Training RMSE: {train_rmse_scores.mean():.4f} mm")
print(f"Average CV Testing RMSE:  {test_rmse_scores.mean():.4f} mm")