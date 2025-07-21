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

df = pd.read_csv("Smart_Farming_Crop_Yield_2024.csv")

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

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        verbosity=0
    ))
])

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

preds = best_model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

print("🎯 Best params:", random_search.best_params_)
print("\n📊 Evaluasi Model XGBoost (Tuned):")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²   : {r2:.4f}")

y_mean = y_train.mean()
baseline_preds = np.full_like(y_test, fill_value=y_mean) 
print("Baseline MAE:", mean_absolute_error(y_test, baseline_preds)) 
print("Baseline R2:", r2_score(y_test, baseline_preds))

from sklearn.model_selection import RandomizedSearchCV 

param_dist_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 9], 
    'learning_rate': [0.01, 0.1, 0.2], 
    'subsample': [0.6, 0.8, 1.0]
}

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)

rand_xgb = RandomizedSearchCV(
    XGBRegressor(objective='reg:squarederror', random_state=42),
    param_dist_xgb, n_iter=20, cv=3, scoring='r2', n_jobs=-1, random_state=42
)

rand_xgb.fit(X_train_enc, y_train)
print("Best XGB params:", rand_xgb.best_params_) 
print("Best XGB R2:", rand_xgb.best_score_)

# Visualizations

plt.figure(figsize=(10, 6))
plt.scatter(y_test, preds, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Predicted vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()

residuals = y_test - preds
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=residuals, color='orange')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals Plot')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

features = num_features + list(best_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out())
importances = best_model.named_steps['regressor'].feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df.head(15), y="Feature", x="Importance", palette="viridis")
plt.title('Top 15 Important Features')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='green')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Save the model
# with open("model-xgboost-tuned.pkl", "wb") as f:
#     pickle.dump(best_model, f)
# print("✅ Model disimpan sebagai model-xgboost-tuned.pkl")
