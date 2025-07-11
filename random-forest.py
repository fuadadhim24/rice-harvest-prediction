import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
import pickle

# === Step 1: Load Dataset ===
df = pd.read_csv("Smart_Farming_Crop_Yield_2024.csv")

# === Step 2: Feature Engineering ===
df["temp_rain_ratio"] = df["temperature_C"] / (df["rainfall_mm"] + 1)
df["moisture_index"] = df["soil_moisture_%"] * df["humidity_%"] / 100

# === Step 3: Pilih Fitur Penting + Kategorikal ===
numeric_features = [
    "soil_moisture_%", "pesticide_usage_ml", "rainfall_mm",
    "temperature_C", "sunlight_hours", "humidity_%",
    "temp_rain_ratio", "moisture_index"
]

categorical_features = ["crop_type", "region", "fertilizer_type"]
df_encoded = pd.get_dummies(df[categorical_features], drop_first=True)

# Gabungkan semua fitur
X = pd.concat([df[numeric_features], df_encoded], axis=1)
y = df["yield_kg_per_hectare"]

# === Step 4: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Step 5: Model Training ===
model = RandomForestRegressor(
    # n_estimators=150,
    # max_depth=10,
    # random_state=42
     n_estimators=300,
    random_state=42,
    n_jobs=-1  # biar cepat
)
model.fit(X_train, y_train)

# === Step 6: Evaluasi ===
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

print("📊 Evaluasi Model (Hybrid Features):")
print(f"MAE : {mae}")
print(f"MSE : {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")

# === Step 7: Simpan Model ===
with open("model-rf-hybrid.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model berhasil disimpan sebagai model-rf-hybrid.pkl")

# === Optional: Visualisasi Feature Importance ===
importances = model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df.head(15), y="Feature", x="Importance", palette="viridis")
plt.title("🔥 Feature Importance - Hybrid Model")
plt.tight_layout()
plt.show()
