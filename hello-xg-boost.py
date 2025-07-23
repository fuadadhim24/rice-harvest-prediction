import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import pickle

# Load dataset
df = pd.read_csv("Smart_Farming_Crop_Yield_2024.csv")

# Filter data where crop_type is "Rice"
df_rice = df[df['crop_type'] == 'Rice']

# Feature lists for rice data
num_features = [
    "soil_moisture_%", "soil_pH", "temperature_C",
    "rainfall_mm", "humidity_%", "sunlight_hours",
    "pesticide_usage_ml", "NDVI_index", "total_days"
]

cat_features = [
    "region", "irrigation_type", "fertilizer_type", "crop_disease_status"
]

# Fill missing categorical values
df_rice[cat_features] = df_rice[cat_features].fillna("Unknown")

# Apply log transformation to target variable
y_rice = np.log1p(df_rice["yield_kg_per_hectare"])

# Prepare the features (X) and target (y)
X_rice = df_rice[num_features + cat_features]

# Split into training and test sets for rice data
X_train_rice, X_test_rice, y_train_rice, y_test_rice = train_test_split(X_rice, y_rice, test_size=0.2, random_state=42)

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# Create the pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        verbosity=0
    ))
])

# Hyperparameter tuning grid for Rice crop prediction
param_grid = {
    "regressor__n_estimators": [100, 150, 200, 300, 500],  # Tambahkan lebih banyak pohon
    "regressor__max_depth": [5, 7, 9, 11],                   # Coba kedalaman pohon lebih tinggi
    "regressor__learning_rate": [0.001, 0.01, 0.05, 0.1],   # Variasikan learning rate
    "regressor__subsample": [0.7, 0.8, 0.9, 1.0],            # Variasikan subset data
    "regressor__colsample_bytree": [0.7, 0.8, 0.9, 1.0]     # Variasikan subset fitur
}

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring="neg_mean_squared_error",  # Kita optimalkan berdasarkan MSE
    verbose=1,
    n_jobs=-1
)

# Fit the model with GridSearchCV
grid_search.fit(X_train_rice, y_train_rice)

# Best model from GridSearchCV
best_model_grid = grid_search.best_estimator_

# Predictions with the best model
preds_grid = best_model_grid.predict(X_test_rice)

# Model evaluation for rice data
mae_grid = mean_absolute_error(y_test_rice, preds_grid)
mse_grid = mean_squared_error(y_test_rice, preds_grid)
rmse_grid = np.sqrt(mse_grid)
r2_grid = r2_score(y_test_rice, preds_grid)

print("🎯 Best params for Rice crop (GridSearch):", grid_search.best_params_)
print("\n📊 Evaluasi Model XGBoost (Tuned) untuk Rice:")
print(f"MAE : {mae_grid:.2f}")
print(f"RMSE: {rmse_grid:.2f}")
print(f"R²   : {r2_grid:.4f}")

# Cross-validation evaluation
cross_val_scores = cross_val_score(best_model_grid, X_rice, y_rice, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validation scores: {cross_val_scores}")
print(f"Mean score: {np.mean(cross_val_scores)}")

# Visualizations

# Predicted vs Actual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test_rice, preds_grid, color='blue', alpha=0.5)
plt.plot([y_test_rice.min(), y_test_rice.max()], [y_test_rice.min(), y_test_rice.max()], 'r--', lw=2)
plt.title('Predicted vs Actual (Rice)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()

# Residuals plot
residuals_grid = y_test_rice - preds_grid
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_rice, y=residuals_grid, color='orange')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals Plot (Rice)')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Feature importance
importances_grid = best_model_grid.named_steps['regressor'].feature_importances_
features_grid = num_features + list(best_model_grid.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out())

feature_importance_df_grid = pd.DataFrame({
    'Feature': features_grid,
    'Importance': importances_grid
})
feature_importance_df_grid = feature_importance_df_grid.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df_grid.head(15), y="Feature", x="Importance", palette="viridis")
plt.title('Top 15 Important Features (Rice)')
plt.show()

# Error distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals_grid, kde=True, color='green')
plt.title('Distribution of Residuals (Rice)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Save the model
with open("model-xgboost-tuned-rice.pkl", "wb") as f:
    pickle.dump(best_model_grid, f)
print("✅ Model disimpan sebagai model-xgboost-tuned-rice.pkl")
