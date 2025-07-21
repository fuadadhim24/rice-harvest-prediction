import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("Smart_Farming_Crop_Yield_2024.csv")

# Fitur
num_features = [
    "soil_moisture_%", "soil_pH", "temperature_C",
    "rainfall_mm", "humidity_%", "sunlight_hours",
    "pesticide_usage_ml", "NDVI_index", "total_days"
]

cat_features = [
    "region", "crop_type", "irrigation_type",
    "fertilizer_type", "crop_disease_status"
]

df[cat_features] = df[cat_features].fillna("Unknown")

X = df[num_features + cat_features]
y = df["yield_kg_per_hectare"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# Pipeline Random Forest
pipeline_rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Hyperparameter tuning
param_grid_rf = {
    "regressor__n_estimators": [100, 150, 200],
    "regressor__max_depth": [5, 10, 15],
    "regressor__min_samples_split": [2, 5],
    "regressor__min_samples_leaf": [1, 2]
}

random_search_rf = RandomizedSearchCV(
    pipeline_rf,
    param_distributions=param_grid_rf,
    n_iter=20,
    cv=5,
    scoring="neg_mean_squared_error",
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Train
random_search_rf.fit(X_train, y_train)
best_model_rf = random_search_rf.best_estimator_

# Save model
with open("best_model_rf.pkl", "wb") as f:
    pickle.dump(best_model_rf, f)

# Save test data
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Optional: evaluasi cepat
preds_rf = best_model_rf.predict(X_test)
print("Model dan data uji berhasil disimpan.")
print(f"MAE : {mean_absolute_error(y_test, preds_rf):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds_rf)):.2f}")
print(f"R²   : {r2_score(y_test, preds_rf):.4f}")

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load model & test data
with open("best_model_rf.pkl", "rb") as f:
    best_model_rf = pickle.load(f)

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()  # jadi Series

# Prediksi
preds_rf = best_model_rf.predict(X_test)
residuals_rf = y_test - preds_rf

# === Visualisasi ===

# 1. Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, preds_rf, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("Gambar 1. RF: Scatter Plot Aktual vs Prediksi")
plt.xlabel("Nilai Aktual (kg/ha)")
plt.ylabel("Prediksi (kg/ha)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Residual Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=residuals_rf, color='orange')
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Gambar 2. RF: Plot Residual")
plt.xlabel("Nilai Aktual")
plt.ylabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Feature Importance
features_rf = best_model_rf.named_steps['preprocessor'].transformers_[0][2] + \
              list(best_model_rf.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out())

importances_rf = best_model_rf.named_steps['regressor'].feature_importances_

feature_importance_df_rf = pd.DataFrame({
    'Feature': features_rf,
    'Importance': importances_rf
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df_rf.head(15), x="Importance", y="Feature", palette="viridis")
plt.title("Gambar 3. RF: Top 15 Fitur Penting")
plt.tight_layout()
plt.show()

# 4. Residual Distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals_rf, kde=True, color='green')
plt.title("Distribusi Residual (Random Forest)")
plt.xlabel("Residuals")
plt.ylabel("Frekuensi")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Tabel Input & Prediksi
sample_input = X_test.iloc[[0]]
sample_pred = best_model_rf.predict(sample_input)

sample_table = sample_input.copy()
sample_table["Estimasi_Hasil_Panen (kg/ha)"] = sample_pred[0]
sample_display = sample_table.T
sample_display.columns = ["Nilai"]

fig, ax = plt.subplots(figsize=(8, len(sample_display) * 0.5 + 1))
ax.axis('off')

table = ax.table(cellText=sample_display.values,
                 rowLabels=sample_display.index,
                 colLabels=["Nilai"],
                 cellLoc='left',
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.title("📋 Tabel Fitur Masukan dan Estimasi Hasil Panen (RF)", pad=20)
plt.tight_layout()
plt.show()
