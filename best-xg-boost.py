import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import joblib

# === Setup ===
os.makedirs("visualizations/xg_boost", exist_ok=True)
os.makedirs("result/xg_boost", exist_ok=True)
os.makedirs("model/xg_boost", exist_ok=True)


# Load data
df = pd.read_csv("Smart_Farming_Crop_Yield_2024.csv")

# Definisikan fitur
num_features = [
    "soil_moisture_%", "soil_pH", "temperature_C",
    "rainfall_mm", "humidity_%", "sunlight_hours",
    "pesticide_usage_ml", "NDVI_index", "total_days"
]

cat_features = [
    "region", "irrigation_type", "fertilizer_type", "crop_disease_status"
]

# === 1. Heatmap Korelasi fitur numerik ===
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap Korelasi Fitur Numerik")
plt.tight_layout()
plt.savefig("visualizations/xg_boost/heatmap_korelasi.png")
plt.close()

# === 2. Distribusi target ===
plt.figure(figsize=(8, 6))
sns.histplot(df["yield_kg_per_hectare"], kde=True, bins=30)
plt.title("Distribusi Yield (kg per hectare)")
plt.xlabel("Yield")
plt.tight_layout()
plt.savefig("visualizations/xg_boost/distribusi_yield.png")
plt.close()

# === 3. Komposisi Crop Type dan Region ===
plt.figure(figsize=(8, 6))
df["crop_type"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Komposisi Crop Type")
plt.xlabel("Crop Type")
plt.tight_layout()
plt.savefig("visualizations/xg_boost/bar_crop_type.png")
plt.close()

plt.figure(figsize=(8, 6))
df["region"].value_counts().plot(kind="pie", autopct='%1.1f%%')
plt.title("Komposisi Region")
plt.tight_layout()
plt.savefig("visualizations/xg_boost/pie_region.png")
plt.close()

# === 4. Missing Value ===
missing = df.isnull().sum()
missing = missing[missing > 0]
if not missing.empty:
    plt.figure(figsize=(10, 6))
    missing.plot(kind="bar", color="salmon")
    plt.title("Missing Values Sebelum Pembersihan")
    plt.tight_layout()
    plt.savefig("visualizations/xg_boost/missing_values.png")
    plt.close()

# === 5. One-Hot Encoding Example ===
sample_cat = df[cat_features].iloc[[0]]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_sample = encoder.fit(df[cat_features]).transform(sample_cat)
encoded_df = pd.DataFrame(encoded_sample, columns=encoder.get_feature_names_out(cat_features))
encoded_df.to_csv("visualizations/xg_boost/sample_encoded.csv", index=False)

# === 6. Boxplot sebelum & sesudah scaling ===
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[num_features])
scaled_df = pd.DataFrame(scaled_data, columns=num_features)

plt.figure(figsize=(12, 6))
sns.boxplot(data=df[num_features])
plt.title("Boxplot Sebelum Scaling")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/xg_boost/boxplot_before_scaling.png")
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=scaled_df)
plt.title("Boxplot Setelah Scaling")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/xg_boost/boxplot_after_scaling.png")
plt.close()

# === 7. Modeling & Evaluasi ===
results = []

for crop in df["crop_type"].unique():
    print(f"\n===== Modeling untuk: {crop} =====")
    subset = df[df["crop_type"] == crop]
    X = subset[num_features + cat_features]
    y = subset["yield_kg_per_hectare"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(objective='reg:squarederror', random_state=42))
    ])

    # Hyperparameter tuning
    param_grid = {
        "regressor__n_estimators": [100, 200, 300, 500],
        "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "regressor__max_depth": [3, 5, 7, 10],
        "regressor__subsample": [0.6, 0.8, 1.0],
        "regressor__colsample_bytree": [0.6, 0.8, 1.0],
        "regressor__gamma": [0, 1, 5],
        "regressor__min_child_weight": [1, 3, 5]
    }



    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=50,
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )



    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    # Save model
    joblib.dump(best_model, f"model/xg_boost/model_{crop}.pkl")

        
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Crop": crop,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    })

    # === Scatter plot per crop ===
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f"Scatter y_test vs prediction - {crop}")
    plt.xlabel("y_test")
    plt.ylabel("prediction")
    plt.tight_layout()
    plt.savefig(f"visualizations/xg_boost/scatter_{crop}.png")
    plt.close()

# Simpan hasil evaluasi
results_df = pd.DataFrame(results)
print("\nHasil Evaluasi per crop_type:")
print(results_df)
results_df.to_csv("result/xg_boost/evaluasi_crop_type_xgboost.csv", index=False)

# === 9. Bar Chart Evaluasi ===
results_df.plot(
    x="Crop",
    y=["MAE", "RMSE", "R2"],
    kind="bar",
    figsize=(12, 6),
    title="Evaluasi Model XGBoost per Crop"
)
plt.ylabel("Nilai")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/xg_boost/bar_hasil_evaluasi.png")
plt.close()

# === 10. Feature Importance ===
importances = best_model.named_steps["regressor"].feature_importances_
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
importances_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importances_df, palette="viridis")
plt.title("Top 20 Feature Importance dari XGBoost")
plt.tight_layout()
plt.savefig("visualizations/xg_boost/feature_importance.png")
plt.close()

print("\nPipeline dan visualisasi selesai. Semua hasil disimpan ke folder 'visualizations/xg_boost' dan result disimpan ke 'result/xg_boost/evaluasi_crop_type_xgboost.csv'.")