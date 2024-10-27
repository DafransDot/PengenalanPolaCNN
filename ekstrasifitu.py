import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Muat dan Preproses Gambar
image_path = 'dataset/penpol1.jpg'  # Ganti dengan path gambar daun jagung kamu
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Ekstraksi Warna dengan Histogram
def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

color_features = extract_color_histogram(image)

# 3. Ekstraksi Tekstur dengan GLCM (Gray Level Co-occurrence Matrix)
def extract_texture_features(gray):
    glcm = cv2.calcHist([gray], [0], None, [256], [0, 256])
    contrast = np.mean(glcm)
    homogeneity = np.sum(glcm) / len(glcm)
    return [contrast, homogeneity]

texture_features = extract_texture_features(gray)

# 4. Ekstraksi Bentuk dengan Hu Moments
def extract_shape_features(gray):
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

shape_features = extract_shape_features(gray)

# 5. Visualisasi Gambar Asli dan Hasil Ekstraksi Fitur
plt.figure(figsize=(15, 5))

# Gambar Asli
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title('Gambar Daun Jagung Asli')
plt.axis('off')

# Gambar Citra Abstraksi Warna
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
plt.title('Representasi Warna (HSV)')
plt.axis('off')

# Gambar Citra Abstraksi Tekstur
plt.subplot(1, 3, 3)
plt.imshow(gray, cmap='gray')
plt.title('Citra Tekstur (Gray Scale)')
plt.axis('off')

plt.tight_layout()
plt.show()

# Tampilkan Fitur yang Diekstraksi
print("Fitur Warna:", color_features[:5])  # Tampilkan beberapa fitur warna
print("Fitur Tekstur:", texture_features)  # Tampilkan fitur tekstur
print("Fitur Bentuk:", shape_features[:5])  # Tampilkan beberapa fitur bentuk
