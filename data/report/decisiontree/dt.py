import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor


csv_path        = "data/report/Geom3_Data.csv" #replace with geometry / dataset used
output_csv      = "data/report/Geom3_Results.csv"
target_column   = "Deflection_mm"
columns_to_drop = ["Run_ID", "Deflection_mm"]

max_trees       = 1000
min_trees       = 50
learning_rate   = 0.05
max_depth       = 3
random_state    = 42

test_size       = 0.25
n_splits        = 10



df = pd.read_csv(csv_path)
X  = df.drop(columns=columns_to_drop)
y  = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

model = GradientBoostingRegressor(
    n_estimators=max_trees,
    learning_rate=learning_rate,
    max_depth=max_depth,
    random_state=random_state,
)
model.fit(X_train, y_train)

staged_train = np.array(list(model.staged_predict(X_train)))
staged_test  = np.array(list(model.staged_predict(X_test)))

n_trees    = np.arange(1, max_trees + 1)
train_rmse = np.sqrt(((staged_train - y_train.values) ** 2).mean(axis=1))
test_rmse  = np.sqrt(((staged_test  - y_test.values)  ** 2).mean(axis=1))

ss_tot_train = ((y_train.values - y_train.values.mean()) ** 2).sum()
ss_tot_test  = ((y_test.values  - y_test.values.mean())  ** 2).sum()
train_r2 = 1 - ((staged_train - y_train.values) ** 2).sum(axis=1) / ss_tot_train
test_r2  = 1 - ((staged_test  - y_test.values)  ** 2).sum(axis=1) / ss_tot_test

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
cv_train_rmse_per_fold = np.zeros((n_splits, max_trees))
cv_test_rmse_per_fold  = np.zeros((n_splits, max_trees))
cv_train_r2_per_fold   = np.zeros((n_splits, max_trees))
cv_test_r2_per_fold    = np.zeros((n_splits, max_trees))

for fold_idx, (tr_idx, te_idx) in enumerate(kfold.split(X)):
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    fold_model = GradientBoostingRegressor(
        n_estimators=max_trees,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    fold_model.fit(X_tr, y_tr)

    pred_tr = np.array(list(fold_model.staged_predict(X_tr)))
    pred_te = np.array(list(fold_model.staged_predict(X_te)))

    cv_train_rmse_per_fold[fold_idx] = np.sqrt(((pred_tr - y_tr.values) ** 2).mean(axis=1))
    cv_test_rmse_per_fold[fold_idx]  = np.sqrt(((pred_te - y_te.values) ** 2).mean(axis=1))

    ss_tot_tr = ((y_tr.values - y_tr.values.mean()) ** 2).sum()
    ss_tot_te = ((y_te.values - y_te.values.mean()) ** 2).sum()
    cv_train_r2_per_fold[fold_idx] = 1 - ((pred_tr - y_tr.values) ** 2).sum(axis=1) / ss_tot_tr
    cv_test_r2_per_fold[fold_idx]  = 1 - ((pred_te - y_te.values)  ** 2).sum(axis=1) / ss_tot_te


cv_train_rmse_mean = cv_train_rmse_per_fold.mean(axis=0)
cv_test_rmse_mean  = cv_test_rmse_per_fold.mean(axis=0)
cv_test_rmse_std   = cv_test_rmse_per_fold.std(axis=0)
cv_train_r2_mean   = cv_train_r2_per_fold.mean(axis=0)
cv_test_r2_mean    = cv_test_r2_per_fold.mean(axis=0)
cv_test_r2_std     = cv_test_r2_per_fold.std(axis=0)

results_df = pd.DataFrame({
    "n_estimators":     n_trees,
    "train_rmse":       train_rmse,
    "test_rmse":        test_rmse,
    "train_r2":         train_r2,
    "test_r2":          test_r2,
    "cv_train_rmse":    cv_train_rmse_mean,
    "cv_test_rmse":     cv_test_rmse_mean,
    "cv_test_rmse_std": cv_test_rmse_std,
    "cv_train_r2":      cv_train_r2_mean,
    "cv_test_r2":       cv_test_r2_mean,
    "cv_test_r2_std":   cv_test_r2_std,
})
results_df = results_df[results_df["n_estimators"] >= min_trees].reset_index(drop=True)
results_df.to_csv(output_csv, index=False)

print(f"\nExported: {output_csv}")

threshold = 1.01 * results_df["test_rmse"].min()
optimal_idx = results_df.index[results_df["test_rmse"] <= threshold][0]
optimal_row = results_df.loc[optimal_idx]
optimal_n   = int(optimal_row["n_estimators"])

print(f"\n--- Optimal n = {optimal_n}) ---")
print(f"  Single-split test RMSE = {optimal_row['test_rmse']:.4f} mm")
print(f"  Single-split train RMSE= {optimal_row['train_rmse']:.4f} mm")
print(f"  Single-split test R^2  = {optimal_row['test_r2']:.4f}")
print(f"  Single-split train R^2 = {optimal_row['train_r2']:.4f}")
print(f"  CV test RMSE           = {optimal_row['cv_test_rmse']:.4f} mm  (+/- {optimal_row['cv_test_rmse_std']:.4f})")
print(f"  CV train RMSE          = {optimal_row['cv_train_rmse']:.4f} mm")
print(f"  CV test R^2            = {optimal_row['cv_test_r2']:.4f}  (+/- {optimal_row['cv_test_r2_std']:.4f})")
print(f"  CV train R^2           = {optimal_row['cv_train_r2']:.4f}")