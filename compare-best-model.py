import matplotlib.pyplot as plt
import pandas as pd

# Data evaluasi model
metrics_df = pd.DataFrame({
    "Model": ["XGBoost", "Random Forest"],
    "MAE (kg/ha)": [1041.70, 1061.24],
    "RMSE (kg/ha)": [1180.11, 1203.36],
    "R²": [-0.0084, -0.0485]
})

# Transpose agar lebih mudah ditampilkan sebagai tabel vertikal
metrics_df.set_index("Model", inplace=True)
metrics_display = metrics_df.T  # transpose

# Visualisasi tabel
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('off')

table = ax.table(
    cellText=metrics_display.values,
    rowLabels=metrics_display.index,
    colLabels=metrics_display.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.title("Tabel: Perbandingan MAE, RMSE, dan R² dari XGBoost dan Random Forest", pad=20)
plt.tight_layout()
plt.show()
