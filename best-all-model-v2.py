import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import os

# Load data
df = pd.read_csv("yield_df.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)

# Filter countries with >= 100 records
country_counts = df['Area'].value_counts()
df = df[df['Area'].isin(country_counts[country_counts >= 100].index)].reset_index(drop=True)

# Encode categorical features
label_encoder = LabelEncoder()
df['Area'] = label_encoder.fit_transform(df['Area'])  # only Area encoded; keep Item as grouping key

# Define models
models = [
    ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor(random_state=42)),
    ('Gradient Boost', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
    ('XGBoost', XGBRegressor(random_state=42)),
    ('KNN', KNeighborsRegressor(n_neighbors=5)),
    ('Decision Tree', DecisionTreeRegressor(random_state=42)),
    ('Bagging Regressor', BaggingRegressor(n_estimators=150, random_state=42))
]

# Final results container
all_results = []

# Loop over each crop type
for item in df['Item'].unique():
    print(f"\n=== Evaluating Models for Crop: {item} ===")
    
    # Filter data by item
    crop_df = df[df['Item'] == item].copy()
    if len(crop_df) < 100:
        print(f"Skipped: Not enough data for {item} ({len(crop_df)} records)")
        continue
    
    X = crop_df.drop(columns=['hg/ha_yield', 'Item'])
    y = crop_df['hg/ha_yield']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Evaluate all models
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc_train = model.score(X_train, y_train)
        acc_test = model.score(X_test, y_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # KFold validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf)
        mean_kfold = np.mean(scores)

        all_results.append({
            'Item': item,
            'Model': name,
            'Train Accuracy': acc_train,
            'Test Accuracy': acc_test,
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'R2 Score': r2,
            'KFold Mean': mean_kfold
        })

        print(f"{name} | Test Accuracy: {acc_test:.4f}, R2: {r2:.4f}, MSE: {mse:.2f}, KFold Mean: {mean_kfold:.4f}")

# Convert to DataFrame
results_df = pd.DataFrame(all_results)
results_df.sort_values(by=['Item', 'Test Accuracy'], ascending=[True, False], inplace=True)


# Buat folder jika belum ada
output_dir = "results/all_models"
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
output_path = os.path.join(output_dir, "model_results_per_item.csv")
results_df.to_csv(output_path, index=False)
print(f"\nEvaluation completed. Results saved to '{output_path}'.")

print(results_df)
