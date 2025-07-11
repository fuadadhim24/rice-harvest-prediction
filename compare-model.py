import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Step 1: Data performa dari beberapa model (isi sesuai hasil evaluasi kamu ya!)
data = {
    "Model": [
        "Random Forest",
        "XGBoost (awal)",
        "XGBoost (tuned)",
        "LightGBM",
        "CatBoost",
        "GradientBoost"
    ],
    "MAE": [1054.56, 1056.75, 1063.85, 1103.00, 1074.17, 1138.90],
    "RMSE": [1203.87, 1262.30, 1200.25, 1291.77, 1228.92, 1300.82],
    "R2": [-0.0494, -0.1537, -0.0431, -0.2082, -0.0935, -0.2252]
}

df_compare = pd.DataFrame(data)

# Step 2: Plot MAE dan RMSE
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(data=df_compare, x="Model", y="MAE", palette="viridis")
plt.xticks(rotation=45)
plt.title("📉 Mean Absolute Error (MAE)")
plt.ylabel("MAE")

plt.subplot(1, 2, 2)
sns.barplot(data=df_compare, x="Model", y="RMSE", palette="magma")
plt.xticks(rotation=45)
plt.title("📉 Root Mean Squared Error (RMSE)")
plt.ylabel("RMSE")

plt.tight_layout()
plt.show()

# Step 3: Plot R² Score
plt.figure(figsize=(8, 4))
sns.barplot(data=df_compare, x="Model", y="R2", palette="coolwarm")
plt.axhline(0, color='gray', linestyle='--')
plt.xticks(rotation=45)
plt.title("📈 R² Score per Model")
plt.ylabel("R² Score")

plt.tight_layout()
plt.show()
