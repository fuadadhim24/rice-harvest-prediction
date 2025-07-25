import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import os


# Load dataset
df = pd.read_csv("yield_df.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)

# Filter countries with >= 100 records
country_counts = df['Area'].value_counts()
df = df[df['Area'].isin(country_counts[country_counts >= 100].index)].reset_index(drop=True)

# Encode Area only
label_encoder = LabelEncoder()
df['Area'] = label_encoder.fit_transform(df['Area'])

# Define model configs with parameter grids
model_configs = [
    ('Linear Regression', LinearRegression(), {}),
    ('Random Forest', RandomForestRegressor(random_state=42), {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }),
    ('Gradient Boost', GradientBoostingRegressor(random_state=42), {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }),
    ('XGBoost', XGBRegressor(random_state=42, verbosity=0), {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }),
    ('KNN', KNeighborsRegressor(), {
        'n_neighbors': [3, 5, 7]
    }),
    ('Decision Tree', DecisionTreeRegressor(random_state=42), {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }),
    ('Bagging Regressor', BaggingRegressor(random_state=42), {
        'n_estimators': [50, 100]
    })
]

# Final results container
all_results = []

# Per Item loop
for item in df['Item'].unique():
    print(f"\n=== Evaluating Models for Crop: {item} ===")
    
    crop_df = df[df['Item'] == item].copy()
    if len(crop_df) < 100:
        print(f"Skipped: Not enough data for {item} ({len(crop_df)} records)")
        continue

    X = crop_df.drop(columns=['hg/ha_yield', 'Item'])
    y = crop_df['hg/ha_yield']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for name, model, param_grid in model_configs:
        print(f"  > Tuning {name}...")
        if param_grid:
            grid = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            best_params = grid.best_params_
        else:
            model.fit(X_train, y_train)
            best_params = "Default"

        y_pred = model.predict(X_test)

        acc_train = model.score(X_train, y_train)
        acc_test = model.score(X_test, y_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # KFold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf)
        mean_kfold = np.mean(scores)

        all_results.append({
            'Item': item,
            'Model': name,
            'Best Params': best_params,
            'Train Accuracy': acc_train,
            'Test Accuracy': acc_test,
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'R2 Score': r2,
            'KFold Mean': mean_kfold
        })

        print(f"    Done: Test R2 = {r2:.4f}, Best Params: {best_params}")

# Buat folder jika belum ada
output_dir = "results/all_models"
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
results_df = pd.DataFrame(all_results)
results_df.sort_values(by=['Item', 'Test Accuracy'], ascending=[True, False], inplace=True)
output_path = os.path.join(output_dir, "model_results_per_item_gridsearch.csv")
results_df.to_csv(output_path, index=False)

print(f"\nEvaluation completed. Results saved to '{output_path}'.")
print(results_df)
