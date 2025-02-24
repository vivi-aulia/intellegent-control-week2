import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# ==========================
# 1. MUAT DAN SIAPKAN DATASET
# ==========================
# Bisa gunakan dataset lain dari Kaggle
dataset_path = "colors_dataset.csv"  # Ganti dengan dataset dari Kaggle jika ada
df = pd.read_csv(dataset_path)

# Pastikan dataset memiliki kolom 'R', 'G', 'B', dan 'color_name'
X = df[['R', 'G', 'B']].values
y = df['color_name'].values

# Split dataset untuk pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
# 2. LATIH MODEL SVM
# ==========================
# GridSearchCV untuk mencari parameter terbaik
param_grid = {'C': [1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1, 1], 'kernel': ['rbf']}
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Simpan model terbaik
best_svm = grid_search.best_estimator_
joblib.dump(best_svm, 'model_svm.pkl')
joblib.dump(scaler, 'scaler.pkl')

# ==========================
# 3. HITUNG AKURASI MODEL
# ==========================
y_pred = best_svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")

# ==========================
# 4. DETEKSI WARNA MULTIPLE DENGAN SVM
# ==========================
# Muat model dan scaler terbaru
try:
    svm = joblib.load('model_svm.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    exit()

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret        print("Error: Could not read frame.")
        break
    
    # Ambil beberapa titik untuk mendeteksi lebih dari satu warna
    height, width, _ = frame.shape
    points = [
        (width // 4, height // 4),  # Kiri atas
        (3 * width // 4, height // 4),  # Kanan atas
        (width // 4, 3 * height // 4),  # Kiri bawah
        (3 * width // 4, 3 * height // 4),  # Kanan bawah
        (width // 2, height // 2)  # Tengah
    ]

    detected_colors = []

    for (x, y) in points:
        roi = frame[y-2:y+3, x-2:x+3]  # Ambil area kecil sekitar titik
        pixel_avg = np.mean(roi, axis=(0, 1)).astype(int)  # Hitung rata-rata RGB
        
        # Normalisasi warna sebelum prediksi
        pixel_scaled = scaler.transform([pixel_avg.reshape(-1)])
        color_pred = svm.predict(pixel_scaled)[0]

        detected_colors.append(color_pred)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Tandai titik

    # Tampilkan warna yang terdeteksi
    colors_text = ", ".join(detected_colors)
    cv2.putText(frame, f'Colors: {colors_text}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Accuracy: {accuracy:.2%}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
