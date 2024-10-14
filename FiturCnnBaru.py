import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Input
import matplotlib.pyplot as plt

# Memuat gambar dan mengubah menjadi array/matriks
img_path = 'dataset/penpol1.jpg'  # Ganti dengan path gambar Anda
img = load_img(img_path, target_size=(200, 200))  # Ubah ukuran gambar jika diperlukan
x = img_to_array(img)  # Mengubah gambar menjadi array/matriks

# Menambah dimensi batch
x = np.expand_dims(x, axis=0)  # Tambah dimensi menjadi (1, 64, 64, 3)

# Membuat input layer
input_img = Input(shape=(200, 200, 3))

# Membangun model CNN untuk ekstraksi fitur
x_model = Conv2D(32, (3, 3), activation='relu')(input_img)
x_model = MaxPooling2D(pool_size=(2, 2))(x_model)
x_model = Conv2D(64, (3, 3), activation='relu')(x_model)
x_model = MaxPooling2D(pool_size=(2, 2))(x_model)
x_model = Conv2D(128, (3, 3), activation='relu')(x_model)
x_model = MaxPooling2D(pool_size=(2, 2))(x_model)

# Membuat model dengan input dan output lapisan konvolusi
model = Model(inputs=input_img, outputs=[x_model])

# Melakukan prediksi pada gambar untuk mendapatkan feature maps
activations = model.predict(x)

# Nama lapisan untuk ditampilkan pada visualisasi
layer_names = ['conv2d_1', 'conv2d_2', 'conv2d_3']

# Mendapatkan dimensi dari feature maps
layer_activation = activations[0]  # Mengambil aktivasi dari layer terakhir

# Visualisasi feature maps untuk lapisan terakhir (Conv2D dengan 128 filter)
num_filters = layer_activation.shape[-1]  # Jumlah filter di lapisan ini
size = layer_activation.shape[1]  # Ukuran feature map

# Membuat grid untuk menampilkan semua filter
n_cols = 8  # Menampilkan 8 filter per baris
n_rows = num_filters // n_cols  # Jumlah baris sesuai dengan jumlah filter

fig, ax = plt.subplots(n_rows, n_cols, figsize=(12, 12))
fig.suptitle("Feature Maps - Last Conv Layer")  # Nama lapisan di atas grid

for i in range(n_rows * n_cols):
    ax[i // n_cols, i % n_cols].imshow(layer_activation[:, :, i], cmap='viridis')
    ax[i // n_cols, i % n_cols].axis('off')  # Matikan axis untuk estetika

plt.show()  # Menampilkan visualisasi
