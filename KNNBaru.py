import os
import cv2
import pandas as pd
import numpy as np

# Fungsi untuk ekstraksi fitur warna (rata-rata RGB)
def extract_features(image_path):
    # Membaca gambar
    image = cv2.imread(image_path)
    
    # Cek jika gambar tidak ditemukan
    if image is None:
        print(f"Error: Gambar tidak ditemukan di {image_path}")
        return None  # Kembalikan None jika gambar tidak ditemukan
    
    # Resize gambar agar ukuran seragam
    image = cv2.resize(image, (128, 128))  # Ubah ukuran gambar menjadi 128x128
    
    # Hitung rata-rata warna RGB
    avg_color = np.mean(image, axis=(0, 1))  # Rata-rata RGB
    return avg_color  # Kembalikan [R, G, B]

# Memuat dataset dari folder DaunSehat dan DaunTidakSehat
def load_dataset(base_path):
    data = []
    labels = []
    
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
            labels.append(folder)  # Menyimpan nama folder sebagai label (DaunSehat atau DaunTidakSehat)
    
    return np.array(data), np.array(labels)

# Tentukan folder dataset
base_path = 'dataset'  # Ganti dengan path folder dataset Anda

# Memuat dataset
X, y = load_dataset(base_path)

# Cek jika data kosong
if X.size == 0 or y.size == 0:
    print("Error: Tidak ada data yang dimuat, periksa folder dataset.")
else:
    # Membuat DataFrame untuk menampilkan tabel
    df = pd.DataFrame({
        "No.": [i+1 for i in range(len(y))],
        "Fitur (R, G, B)": [str(features) for features in X],
        "Label": y
    })

    # Menampilkan tabel data
    print("Tabel Daun Sehat dan Tidak Sehat:")
    print(df)
