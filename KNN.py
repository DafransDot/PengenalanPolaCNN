import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Fungsi untuk ekstraksi fitur warna (rata-rata RGB)
def extract_features(image_path):
    # Membaca gambar
    image = cv2.imread(image_path)
    
    # Cek jika gambar tidak ditemukan
    if image is None:
        print(f"Error: Gambar tidak ditemukan di {image_path}")
        return None  # Kembalikan None jika gambar tidak ditemukan
    
    # Resize gambar agar ukuran seragam
    image = cv2.resize(image, (128, 128))
    # Hitung rata-rata warna RGB
    avg_color = np.mean(image, axis=(0, 1))  # Rata-rata RGB
    return avg_color  # Kembalikan [R, G, B]

# Memuat dataset dari folder
def load_dataset(base_path):
    data = []
    labels = []
    
    # Cek apakah folder dasar ada
    if not os.path.exists(base_path):
        print(f"Error: Folder {base_path} tidak ditemukan.")
        return np.array([]), np.array([])

    # Loop untuk folder DaunSehat dan DaunTidakSehat
    for label, folder in enumerate(['DaunSehat', 'DaunTidakSehat']):  # 0: sehat, 1: tidak sehat
        folder_path = os.path.join(base_path, folder)
        
        # Cek apakah folder ada
        if not os.path.exists(folder_path):
            print(f"Error: Folder {folder_path} tidak ditemukan.")
            continue
        
        # Loop untuk setiap file gambar di dalam folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Ekstraksi fitur dari gambar
            features = extract_features(file_path)
            
            # Jika fitur tidak ada (gambar tidak ditemukan), lewati gambar ini
            if features is None:
                continue
            
            data.append(features)
            labels.append(label)
    
    return np.array(data), np.array(labels)

# Fungsi untuk memprediksi gambar baru
def predict_image(image_path, model, scaler):
    features = extract_features(image_path)  # Ekstraksi fitur dari gambar
    if features is None:
        return None
    
    # Normalisasi fitur
    features = scaler.transform([features])  # Normalisasi seperti data training
    prediction = model.predict(features)
    
    return "Sehat" if prediction[0] == 0 else "Tidak Sehat"

# Load data
base_path = 'dataset'  # Ganti dengan path folder dataset Anda
X, y = load_dataset(base_path)

# Cek jika data kosong
if X.size == 0 or y.size == 0:
    print("Error: Tidak ada data yang dimuat, periksa folder dataset.")
else:
    # Split data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalisasi data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Inisialisasi dan pelatihan model KNN
    k = 1  # Jumlah tetangga terdekat
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Prediksi pada data uji
    y_pred = knn.predict(X_test)

    # Evaluasi Model
    print("Akurasi Model:", accuracy_score(y_test, y_pred))
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred, target_names=["Sehat", "Tidak Sehat"]))

    # Membuat tabel klasifikasi
    hasil = pd.DataFrame({
        "Data Uji": [f"Data {i+1}" for i in range(len(y_test))],
        "Fitur": list(X_test),
        "Prediksi": ["Sehat" if pred == 0 else "Tidak Sehat" for pred in y_pred],
        "Asli": ["Sehat" if actual == 0 else "Tidak Sehat" for actual in y_test]
    })

    print("\nTabel Klasifikasi:")
    print(hasil)

    # Pilih gambar untuk diuji
    image_path = input("Masukkan path gambar daun yang akan diuji (misalnya 'dataset/DaunSehat/daun1.jpg'): ")
    result = predict_image(image_path, knn, scaler)
    
    if result:
        print(f"Hasil prediksi untuk gambar: {result}")
