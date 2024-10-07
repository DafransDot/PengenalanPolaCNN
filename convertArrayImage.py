import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Path gambar contoh
img_path = 'dataset/penpol1.jpg'  # Pastikan path gambar benar

# Memuat gambar
img = load_img(img_path, target_size=(64, 64))  # Memuat gambar dan mengubah ukuran

# Mengubah gambar menjadi array/matriks
x = img_to_array(img)  # Mengubah gambar menjadi array/matriks
print("Bentuk matriks gambar:", x.shape)  # Menampilkan ukuran matriks (64, 64, 3)

# Agar numpy menampilkan semua elemen array tanpa dipotong
np.set_printoptions(threshold=np.inf)

# Menampilkan semua nilai dari matriks gambar
print("Isi matriks gambar:")
print(x)
