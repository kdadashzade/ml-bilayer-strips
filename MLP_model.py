import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 1
# Dataset from ABAQUS is imported and read
df = pd.read_csv('bimetal_dataset.csv')

# X values = Independent variables
# Y values = Dependent variables
X = df[['Length_mm', 'Active_Fraction', 'Temp_Change_C']]
y = df['Deflection_mm']

# 2
# Splits data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Pre-Processing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Building ML Model
nn_model = MLPRegressor(hidden_layer_sizes=(64, 64),
                        activation='relu',
                        solver='adam',
                        max_iter=5000,
                        random_state=50)

print("Training Neural Network...")
nn_model.fit(X_train_scaled, y_train)

# 5. EVALUATE
# Note: We must pass the SCALED test data, not the raw data
predictions = nn_model.predict(X_test_scaled)
score = r2_score(y_test, predictions)

print(f"Neural Network Accuracy (R²): {score:.4f}")

# 6. PREDICT NEW VALUES
# ---------------------
# Instead of a list, create a tiny DataFrame with the SAME column names
new_data = pd.DataFrame([[100.0, 0.5, 100.0]],
                        columns=['Length_mm', 'Active_Fraction', 'Temp_Change_C'])

# Now the scaler won't complain because the names match!
new_data_scaled = scaler.transform(new_data)

new_pred = nn_model.predict(new_data_scaled)
print(f"Predicted Deflection: {new_pred[0]:.2f} mm")