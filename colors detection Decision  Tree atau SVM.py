import cv2
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Muat model Decision Tree dan scaler
try:
    dt_model = joblib.load('model_knn.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    exit()

# Inisialisasi kamera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    height, width, _ = frame.shape

    # Contoh: deteksi warna pixel tengah
    pixel_center = frame[height // 2, width // 2]

    # Normalisasi pixel sebelum prediksi
    pixel_center_scaled = scaler.transform([pixel_center])

    # Prediksi warna menggunakan model Decision Tree
    color_pred = dt_model.predict(pixel_center_scaled)[0]

    # Deteksi beberapa warna (contoh: merah dan biru)
    red_threshold = 150  # Contoh threshold untuk warna merah
    blue_threshold = 100  # Contoh threshold untuk warna biru

    red_mask = (frame[:, :, 2] > red_threshold) & (frame[:, :, 0] < blue_threshold) & (frame[:, :, 1] < blue_threshold)
    blue_mask = (frame[:, :, 0] > blue_threshold) & (frame[:, :, 1] < red_threshold) & (frame[:, :, 2] < red_threshold)

    # Tampilkan warna yang terdeteksi pada frame
    if red_mask.any():
        cv2.putText(frame, 'Red', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if blue_mask.any():
        cv2.putText(frame, 'Blue', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()