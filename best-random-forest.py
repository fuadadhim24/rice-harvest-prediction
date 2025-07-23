import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup direktori
os.makedirs("model/random_forest", exist_ok=True)
os.makedirs("result/random_forest", exist_ok=True)
os.makedirs("visualizations/random_forest", exist_ok=True)

# Load data
df = pd.read_csv("Smart_Farming_Crop_Yield_2024.csv")

# Fitur
num_features = [
    "soil_moisture_%", "soil_pH", "temperature_C",
    "rainfall_mm", "humidity_%", "sunlight_hours",
    "pesticide_usage_ml", "NDVI_index", "total_days"
]

cat_features = [
    "region", "irrigation_type", "fertilizer_type", "crop_disease_status"
]

# Visualisasi missing value sebelum pembersihan
missing = df[cat_features + ["crop_type"]].isnull().sum()
plt.figure(figsize=(10, 5))
sns.barplot(x=missing.index, y=missing.values)
plt.title("Missing Value Sebelum Pembersihan")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/random_forest/missing_value_before_cleaning.png")
plt.close()

# Bersihkan missing value kategorikal
df[cat_features + ["crop_type"]] = df[cat_features + ["crop_type"]].fillna("Unknown")

# Heatmap Korelasi fitur numerik
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_features + ["yield_kg_per_hectare"]].corr(), annot=True, cmap="coolwarm")
plt.title("Heatmap Korelasi Fitur Numerik")
plt.tight_layout()
plt.savefig("visualizations/random_forest/heatmap_correlation.png")
plt.close()

# Distribusi yield_kg_per_hectare
plt.figure(figsize=(8, 5))
sns.histplot(df["yield_kg_per_hectare"], kde=True)
plt.title("Distribusi Target: yield_kg_per_hectare")
plt.xlabel("Yield (kg/hectare)")
plt.tight_layout()
plt.savefig("visualizations/random_forest/target_distribution.png")
plt.close()

# Komposisi crop_type (barchart)
plt.figure(figsize=(10, 5))
df["crop_type"].value_counts().plot(kind="bar")
plt.title("Komposisi crop_type")
plt.xlabel("Crop Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("visualizations/random_forest/crop_type_distribution.png")
plt.close()

# Komposisi region (pie chart)
plt.figure(figsize=(6, 6))
df["region"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Komposisi Region")
plt.tight_layout()
plt.savefig("visualizations/random_forest/region_distribution_pie.png")
plt.close()

# Simpan hasil evaluasi
results = []

# Looping per crop_type
for crop in df["crop_type"].unique():
    print(f"\n===== Modeling untuk: {crop} =====")
    subset = df[df["crop_type"] == crop]

    X = subset[num_features + cat_features]
    y = subset["yield_kg_per_hectare"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ])

    # Pipeline Random Forest
    pipeline_rf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ])

    # Hyperparameter tuning
    param_grid_rf = {
        "regressor__n_estimators": [100, 150],
        "regressor__max_depth": [5, 10],
        "regressor__min_samples_split": [2, 5],
        "regressor__min_samples_leaf": [1, 2]
    }

    random_search_rf = RandomizedSearchCV(
        pipeline_rf,
        param_distributions=param_grid_rf,
        n_iter=5,
        cv=3,
        scoring="neg_mean_squared_error",
        verbose=0,
        random_state=42,
        n_jobs=-1
    )

    # Train
    random_search_rf.fit(X_train, y_train)
    best_model = random_search_rf.best_estimator_

    # Save model
    with open(f"model/random_forest/best_model_rf_{crop}.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Predict dan evaluasi
    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Visualisasi y_test vs prediction
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.title(f"Prediction Accuracy: {crop}")
    plt.tight_layout()
    plt.savefig(f"visualizations/random_forest/y_test_vs_pred_{crop}.png")
    plt.close()

    # Visualisasi Feature Importance
    rf_model = best_model.named_steps["regressor"]
    ohe = best_model.named_steps["preprocessor"].named_transformers_["cat"]
    feature_names = num_features + list(ohe.get_feature_names_out(cat_features))
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-10:]  # top 10

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.title(f"Top 10 Feature Importance: {crop}")
    plt.tight_layout()
    plt.savefig(f"visualizations/random_forest/feature_importance_{crop}.png")
    plt.close()

    results.append({
        "Crop": crop,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    })

# Tampilkan hasil evaluasi
results_df = pd.DataFrame(results)
print("\nHasil Evaluasi per crop_type:")
print(results_df)

# Simpan evaluasi
results_df.to_csv("result/random_forest/evaluasi_crop_type_random_forest.csv", index=False)

# Visualisasi hasil evaluasi model
plt.figure(figsize=(12, 5))
metrics = ["MAE", "RMSE", "R2"]
for i, metric in enumerate(metrics, 1):
    plt.subplot(1, 3, i)
    sns.barplot(data=results_df, x="Crop", y=metric)
    plt.title(f"{metric} per Crop")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("visualizations/random_forest/evaluation_metrics_per_crop.png")
plt.close()

# Contoh One-Hot Encoding hasil transformasi
sample_encoded = preprocessor.fit_transform(df[num_features + cat_features].iloc[:5])
encoded_df = pd.DataFrame(sample_encoded)
encoded_df.to_csv("visualizations/random_forest/sample_encoded.csv", index=False)

print("\nPipeline dan visualisasi selesai. Semua hasil disimpan ke folder 'visualizations/random_forest'")
