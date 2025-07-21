# Smart Farming Crop Yield Prediction

Proyek ini bertujuan untuk memprediksi hasil panen padi berdasarkan data sensor lapangan menggunakan algoritma Machine Learning: XGBoost dan Random Forest.

Dataset yang digunakan berasal dari Kaggle:
📦 https://www.kaggle.com/datasets/atharvasoundankar/smart-farming-sensor-data-for-yield-prediction/data

---

## 🔧 Cara Penggunaan

### 1. Menjalankan Model Terbaik XGBoost
File: best-xg-boost.py

Menjalankan pipeline XGBoost dengan hyperparameter terbaik, disertai evaluasi visual dan penyimpanan model.

Cara menjalankan:
> python best-xg-boost.py

Output:
- Evaluasi performa: MAE, RMSE, R²
- Scatter plot aktual vs prediksi
- Visualisasi residual error
- Feature importance
- Model disimpan ke file: model-xgboost-tuned.pkl

---

### 2. Menjalankan Model Terbaik Random Forest
File: best-random-forest.py

Menjalankan pipeline Random Forest dengan hasil tuning terbaik, termasuk visualisasi dan penyimpanan model.

Cara menjalankan:
> python best-random-forest.py

Output:
- Evaluasi performa dan visualisasi prediksi
- Model disimpan ke file: best_model_rf.pkl

---

### 3. Membandingkan Model Terbaik
File: compare-best-model.py

Membandingkan performa model terbaik (XGBoost vs Random Forest).

Cara menjalankan:
> python compare-best-model.py

Output:
- Visualisasi perbandingan performa kedua model
- Analisis model yang paling optimal

---

## 📁 Dataset dan Data Uji

- Smart_Farming_Crop_Yield_2024.csv  → Dataset utama
- X_test.csv, y_test.csv             → Data uji untuk evaluasi final

---

## 📦 Instalasi Dependensi

Pastikan telah menginstal pustaka berikut:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn

---

© 2024 Fuad Adhim
