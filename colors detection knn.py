import cv2
import joblib
import numpy as np

# Muat model KNN dan scaler
knn = joblib.load('model_knn.pkl')
scaler = joblib.load('scaler.pkl')

# Muat dataset uji jika tersedia
X_test = joblib.load('X_test.pkl')  # Data fitur uji
y_test = joblib.load('y_test.pkl')  # Label uji

# Normalisasi data uji
X_test_scaled = scaler.transform(X_test)

# Hitung akurasi model
accuracy = knn.score(X_test_scaled, y_test)
print(f"Model Accuracy: {accuracy:.2%}")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Ambil pixel tengah gambar
    height, width, _ = frame.shape
    pixel_center = frame[height//2, width//2]
    
    # Normalisasi pixel sebelum prediksi
    pixel_center_scaled = scaler.transform([pixel_center.reshape(-1)])
    
    # Prediksi warna
    color_pred = knn.predict(pixel_center_scaled)[0]
    
    # Tampilkan warna dan akurasi pada frame
    cv2.putText(frame, f'Color: {color_pred}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Accuracy: {accuracy:.2%}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()