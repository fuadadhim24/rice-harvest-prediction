import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import pickle

# === Step 1: Load Dataset ===
df = pd.read_csv("Smart_Farming_Crop_Yield_2024.csv")

# === Step 2: Definisikan Fitur ===
num_features = [
    "soil_moisture_%", "soil_pH", "temperature_C",
    "rainfall_mm", "humidity_%", "sunlight_hours",
    "pesticide_usage_ml", "NDVI_index", "total_days"
]

cat_features = [
    "region", "crop_type", "irrigation_type",
    "fertilizer_type", "crop_disease_status"
]

# === Step 3: Pra-pemrosesan ===
df[cat_features] = df[cat_features].fillna("Unknown")

X = df[num_features + cat_features]
y = df["yield_kg_per_hectare"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Step 4: Pipeline Preprocessing ===
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# === Step 5: Definisikan Pipeline Model ===
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        verbosity=0
    ))
])

# === Step 6: Hyperparameter Tuning ===
param_grid = {
    "regressor__n_estimators": [100, 150, 200],
    "regressor__max_depth": [3, 5, 7],
    "regressor__learning_rate": [0.01, 0.05, 0.1],
    "regressor__subsample": [0.7, 0.8, 1.0],
    "regressor__colsample_bytree": [0.7, 0.8, 1.0],
}

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=100,
    cv=5,
    scoring="neg_mean_squared_error",
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

print("🎯 Best params:", random_search.best_params_)

# === Step 7: Evaluasi ===
preds = best_model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

print("\n📊 Evaluasi Model XGBoost (Tuned):")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# === Step 8: Simpan Model ===
with open("model-xgboost-tuned.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("✅ Model disimpan sebagai model-xgboost-tuned.pkl")
