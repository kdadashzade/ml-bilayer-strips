import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

print("--- Running Master Evaluation Script ---")

# ---------------------------------------------------------
# 1. LOAD DATA & APPLY ENGINEERING FILTER
# ---------------------------------------------------------
csv_file = 'Bimetal_Dataset_200Runs_1.csv'
df = pd.read_csv(csv_file)

# Calculate Slenderness Ratio and filter out "noodles"
df['Slenderness'] = df['Length_mm'] / (df['TPU_Thick_mm'] + df['PLA_Thick_mm'])
df_clean = df[df['Slenderness'] < 400].copy()
print(f"Removed {len(df) - len(df_clean)} extreme 'noodle' geometry(s).\n")

# Prepare Features & Target
X = df_clean[['Length_mm', 'TPU_Thick_mm', 'PLA_Thick_mm']]
y = df_clean['Deflection_mm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=100)


# 2. PART A: COMPARE POLYNOMIAL DEGREES
# ---------------------------------------------------------
degrees_to_test = [1, 2, 3, 4, 5]
train_scores = []
test_scores = []

for d in degrees_to_test:
    temp_model = make_pipeline(StandardScaler(), PolynomialFeatures(d), Ridge(alpha=1.0))
    temp_model.fit(X_train, y_train)

    train_scores.append(r2_score(y_train, temp_model.predict(X_train)))
    test_scores.append(r2_score(y_test, temp_model.predict(X_test)))

    print(f"Degree {d} -> Train R2: {train_scores[-1]:.4f} | Test R2: {test_scores[-1]:.4f}")

# Plot 1: Validation Curve
plt.figure(1, figsize=(9, 5))
plt.plot(degrees_to_test, train_scores, marker='o', color='green', label='Training Accuracy', linewidth=2)
plt.plot(degrees_to_test, test_scores, marker='s', color='blue', label='Testing Accuracy', linewidth=2)
plt.title('Figure 1: Validation Curve (Finding the Sweet Spot)')
plt.xlabel('Polynomial Degree (Complexity)')
plt.ylabel('R-squared Score')
plt.xticks(degrees_to_test)
best_degree = degrees_to_test[np.argmax(test_scores)]
plt.axvline(x=best_degree, color='red', linestyle='--', label=f'Best Test Degree: {best_degree}')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# ---------------------------------------------------------
# 3. PART B: EVALUATE THE BEST MODEL (DEGREE 2 RIDGE)
# ---------------------------------------------------------
print("\n--- Training Final Degree 2 Ridge Model ---")
final_model = make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1.0))
final_model.fit(X_train, y_train)

y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

# Plot 2: Actual vs. Predicted
plt.figure(2, figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.7, label='Test Data')
plt.scatter(y_train, y_train_pred, color='green', alpha=0.3, label='Train Data')

min_val = min(y.min(), min(y_test_pred), min(y_train_pred))
max_val = max(y.max(), max(y_test_pred), max(y_train_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Fit')

plt.title('Figure 2: Actual vs. Predicted Deflection (Degree 2 Ridge)')
plt.xlabel('Actual Deflection (mm)')
plt.ylabel('Predicted Deflection (mm)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Plot 3: Residuals
plt.figure(3, figsize=(8, 6))
plt.scatter(y_train_pred, y_train - y_train_pred, color='green', alpha=0.4, label='Train Residuals')
plt.scatter(y_test_pred, y_test - y_test_pred, color='blue', alpha=0.6, label='Test Residuals')

plt.hlines(y=0, xmin=min(y_test_pred.min(), y_train_pred.min()),
           xmax=max(y_test_pred.max(), y_train_pred.max()), color='red', linestyle='--')

plt.title('Figure 3: Residual Error Distribution (Degree 2 Ridge)')
plt.xlabel('Predicted Deflection (mm)')
plt.ylabel('Residual Error (Actual - Predicted) (mm)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Show all three graphs at the same time
plt.show()