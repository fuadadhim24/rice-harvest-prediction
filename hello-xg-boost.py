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

# === Step 2: Feature Engineering ===
df["temp_x_sun"] = df["temperature_C"] * df["sunlight_hours"]
df["humidity_ratio"] = df["humidity_%"] / (df["temperature_C"] + 1e-5)
df["rain_div_days"] = df["rainfall_mm"] / (df["total_days"] + 1e-5)

# === Step 3: Definisikan fitur numerik & kategorik ===
num_features = [
    "soil_moisture_%", "soil_pH", "temperature_C", "rainfall_mm", "humidity_%",
    "sunlight_hours", "pesticide_usage_ml", "NDVI_index", "total_days",
    "temp_x_sun", "humidity_ratio", "rain_div_days"
]

cat_features = ["region", "crop_type", "irrigation_type", "fertilizer_type", "crop_disease_status"]

# Handle missing
df[cat_features] = df[cat_features].fillna("Unknown")

# === Step 4: Split Data ===
X = df[num_features + cat_features]
y = df["yield_kg_per_hectare"]
y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# === Step 5: Preprocessing Pipeline ===
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# === Step 6: XGBoost Pipeline ===
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0))
])

# === Step 7: Hyperparameter Tuning ===
param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [4, 6, 8],
    "regressor__learning_rate": [0.01, 0.03],
    "regressor__subsample": [0.7, 0.9],
    "regressor__colsample_bytree": [0.7, 1.0],
}

random_search = RandomizedSearchCV(
    pipeline, param_distributions=param_grid,
    n_iter=50, cv=5, scoring="neg_mean_squared_error",
    verbose=1, random_state=42, n_jobs=-1
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
print("🎯 Best params:", random_search.best_params_)

# === Step 8: Evaluasi ===
preds_log = best_model.predict(X_test)
preds = np.expm1(preds_log)
y_test_actual = np.expm1(y_test)

mae = mean_absolute_error(y_test_actual, preds)
rmse = np.sqrt(mean_squared_error(y_test_actual, preds))
r2 = r2_score(y_test_actual, preds)

print("\n📊 Evaluasi Model XGBoost (Log + Engineered):")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# === Step 9: Save Model ===
with open("model-xgboost-log-eng.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("✅ Model disimpan sebagai model-xgboost-log-eng.pkl")

# === Step 10: Feature Importance ===
xgb_model = best_model.named_steps["regressor"]
pre = best_model.named_steps["preprocessor"]
num_cols = pre.named_transformers_["num"].feature_names_in_
cat_cols = pre.named_transformers_["cat"].get_feature_names_out(cat_features)
feat_names = list(num_cols) + list(cat_cols)
importances = xgb_model.feature_importances_

feat_imp_df = pd.DataFrame({
    "Feature": feat_names,
    "Importance": importances
}).sort_values("Importance", ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df, y="Feature", x="Importance", palette="viridis")
plt.title("🔥 Top 20 Fitur Terpenting")
plt.tight_layout()
plt.show()
