import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
df = pd.read_csv("Smart_Farming_Crop_Yield_2024.csv")

# List of numerical and categorical features
num_features = [
    "soil_moisture_%", "soil_pH", "temperature_C",
    "rainfall_mm", "humidity_%", "sunlight_hours",
    "pesticide_usage_ml", "NDVI_index", "total_days"
]

cat_features = [
    "region", "crop_type", "irrigation_type",
    "fertilizer_type", "crop_disease_status"
]

# Fill missing categorical data with "Unknown"
df[cat_features] = df[cat_features].fillna("Unknown")

# Prepare features (X) and target (y)
X = df[num_features + cat_features]
y = df["yield_kg_per_hectare"]

# Log transform the target variable (if needed)
# y = np.log1p(y)  # Uncomment this line if you want to apply log transformation

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# Define the Random Forest pipeline
pipeline_rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Hyperparameter grid for RandomizedSearchCV
param_grid_rf = {
    "regressor__n_estimators": [100, 200, 300],
    "regressor__max_depth": [3, 5, 7, 9],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 4],
}

# Set up RandomizedSearchCV for Random Forest
random_search_rf = RandomizedSearchCV(
    pipeline_rf,
    param_distributions=param_grid_rf,
    n_iter=50,
    cv=5,
    scoring="neg_mean_squared_error",
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit the Random Forest model
random_search_rf.fit(X_train, y_train)

# Best model and parameters
best_model_rf = random_search_rf.best_estimator_
print("🎯 Best params for Random Forest:", random_search_rf.best_params_)

# Predictions on the test set
preds_rf = best_model_rf.predict(X_test)

# Evaluation metrics
mae_rf = mean_absolute_error(y_test, preds_rf)
mse_rf = mean_squared_error(y_test, preds_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, preds_rf)

# Print evaluation results for Random Forest
print("\n📊 Evaluasi Model Random Forest:")
print(f"MAE : {mae_rf:.2f}")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R²   : {r2_rf:.4f}")

# Optional: Save the best model
# with open("model-random-forest-tuned.pkl", "wb") as f:
#     pickle.dump(best_model_rf, f)
# print("✅ Model disimpan sebagai model-random-forest-tuned.pkl")
