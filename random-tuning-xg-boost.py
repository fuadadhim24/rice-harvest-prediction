import os
import json
import random
import joblib
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# === Konfigurasi visualisasi ===
mpl.rcParams['figure.dpi'] = 100
warnings.filterwarnings("ignore")

# === Load dataset ===
df = pd.read_csv("Smart_Farming_Crop_Yield_2024.csv")

# === Fitur numerik dan kategorikal ===
num_features = [
    "soil_moisture_%", "soil_pH", "temperature_C",
    "rainfall_mm", "humidity_%", "sunlight_hours",
    "pesticide_usage_ml", "NDVI_index", "total_days"
]

cat_features = [
    "region", "irrigation_type", "fertilizer_type", "crop_disease_status"
]

# === Konfigurasi model dan parameter grid ===
models_config = {
    "XGBoost": {
        "model": XGBRegressor(objective='reg:squarederror', random_state=42),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0, 1, 5],
            "min_child_weight": [1, 3, 5]
        }
    },
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    },
    "DecisionTree": {
        "model": DecisionTreeRegressor(random_state=42),
        "param_grid": {
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    },
    "LinearRegression": {
        "model": LinearRegression(),
        "param_grid": {}
    },
    "GradientBoosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "param_grid": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    }
}

# === Fungsi tuning acak ===
def manual_model_search(X_train, X_test, y_train, y_test, model, param_grid, preprocessor, max_iter=20, stop_r2=0.3):
    best_r2 = -np.inf
    best_model = None
    best_params = None
    log = []

    param_list = list(ParameterSampler(param_grid, n_iter=max_iter, random_state=42)) if param_grid else [{}]

    for i, params in enumerate(param_list):
        reg = clone(model).set_params(**params)
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", reg)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        log.append({
            "iteration": i + 1,
            "params": params,
            "R2": round(r2, 4)
        })

        if r2 > best_r2:
            best_r2 = r2
            best_model = pipe
            best_params = params

        if r2 >= stop_r2:
            break

    return best_model, best_params, best_r2, log

# === Proses modeling per jenis tanaman & model ===
all_results = []

for crop in df["crop_type"].unique():
    print(f"\n=== 🚜 Crop: {crop} ===")
    subset = df[df["crop_type"] == crop]
    X = subset[num_features + cat_features]
    y = subset["yield_kg_per_hectare"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    for model_name, config in models_config.items():
        print(f"\n🔍 Model: {model_name}")
        base_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", clone(config["model"]))
        ])

        base_pipe.fit(X_train, y_train)
        y_pred_init = base_pipe.predict(X_test)
        r2_init = r2_score(y_test, y_pred_init)
        print(f"   • R² awal = {r2_init:.4f}")

        # Siapkan folder per model
        model_result_dir = f"result/xg_boost/per_crop/{model_name}"
        model_model_dir = f"model/xg_boost/per_crop/{model_name}"
        os.makedirs(model_result_dir, exist_ok=True)
        os.makedirs(model_model_dir, exist_ok=True)

        # Jika R2 negatif, tuning dilakukan
        if r2_init < 0:
            print("   ⚙️  Tuning dilakukan...")
            best_model, best_params, best_r2, log = manual_model_search(
                X_train, X_test, y_train, y_test,
                config["model"], config["param_grid"],
                preprocessor
            )
            pd.DataFrame(log).to_csv(f"{model_result_dir}/tuning_log_{crop}.csv", index=False)
            with open(f"{model_result_dir}/best_params_{crop}.json", "w") as f:
                json.dump(best_params, f, indent=4)
        else:
            best_model = base_pipe
            best_params = config["model"].get_params() if hasattr(config["model"], "get_params") else {}

        # Evaluasi akhir
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        all_results.append({
            "Crop": crop,
            "Model": model_name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4)
        })

        # Simpan model
        model_path = f"{model_model_dir}/model_{crop}.pkl"
        joblib.dump(best_model, model_path)

# === Simpan hasil keseluruhan evaluasi ===
final_df = pd.DataFrame(all_results)
final_result_path = "result/xg_boost/per_crop/all_model_comparison.csv"
final_df.to_csv(final_result_path, index=False)
print(f"\n✅ Semua hasil evaluasi model disimpan dalam '{final_result_path}'")
