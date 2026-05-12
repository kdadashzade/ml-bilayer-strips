import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline


GEOMETRY_LABEL = 'Geometry 3'
csv_path       = 'data/report/Geom3_Data.csv'
output_csv     = 'data/report/Geom3_PolyResults.csv'

target_column   = 'Deflection_mm'
feature_columns = ['Length_mm', 'PLA_Thick_mm', 'TPU_Thick_mm', 'Shrinkage_Strain']

test_size       = 0.25
random_state    = 42
n_splits        = 10
cv_random_state = 42
ORDERS          = range(1, 8)


df = pd.read_csv(csv_path)
X = df[feature_columns]
y = df[[target_column]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)


def compute_bic(y_true, y_pred, n_params):
    n = len(y_true)
    rss = np.sum((np.ravel(y_true) - np.ravel(y_pred)) ** 2)
    return n * np.log(rss / n) + n_params * np.log(n)


def evaluate_polynomial_order(degree, X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('model', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred  = pipeline.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse  = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2   = r2_score(y_train, y_train_pred)
    test_r2    = r2_score(y_test, y_test_pred)

    n_features = pipeline.named_steps['poly'].n_output_features_
    n_params   = n_features + 1
    bic        = compute_bic(y_train.values, y_train_pred, n_params)

    return {
        'degree':     degree,
        'n_params':   n_params,
        'train_rmse': train_rmse,
        'test_rmse':  test_rmse,
        'train_r2':   train_r2,
        'test_r2':    test_r2,
        'bic':        bic,
        'pipeline':   pipeline,
    }

results = [evaluate_polynomial_order(d, X_train, X_test, y_train, y_test)
           for d in ORDERS]

results_df = pd.DataFrame([
    {k: v for k, v in r.items() if k != 'pipeline'} for r in results
])

results_df.to_csv(output_csv, index=False)
print(f"\nExported: {output_csv}")


best_idx      = results_df['bic'].idxmin()
BEST_DEGREE   = int(results_df.loc[best_idx, 'degree'])
best_result   = results[best_idx]
best_pipeline = best_result['pipeline']
intercept_val = np.ravel(best_pipeline.named_steps['model'].intercept_)[0]

cv_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=BEST_DEGREE, include_bias=False)),
    ('model', LinearRegression())
])

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=cv_random_state)
cv_results = cross_validate(
    cv_pipeline, X, y,
    cv=kfold,
    scoring=['neg_mean_squared_error', 'r2'],
    return_train_score=True
)

cv_train_rmse = np.sqrt(-cv_results['train_neg_mean_squared_error'])
cv_test_rmse  = np.sqrt(-cv_results['test_neg_mean_squared_error'])
cv_train_r2   = cv_results['train_r2']
cv_test_r2    = cv_results['test_r2']


print(f"  Single-split test RMSE = {best_result['test_rmse']:.4f} mm")
print(f"  Single-split train RMSE= {best_result['train_rmse']:.4f} mm")
print(f"  Single-split test R^2  = {best_result['test_r2']:.4f}")
print(f"  Single-split train R^2 = {best_result['train_r2']:.4f}")
print(f"  CV test RMSE = {cv_test_rmse.mean():.4f} mm")
print(f"  CV train RMSE = {cv_train_rmse.mean():.4f} mm")
print(f"  CV test R^2 = {cv_test_r2.mean():.4f}")
print(f"  CV train R^2 = {cv_train_r2.mean():.4f}")