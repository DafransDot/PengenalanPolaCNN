import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Membuat dan melatih model CNN
def buat_model_cnn():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Melatih model CNN
def latih_model_cnn(model):
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_set = train_datagen.flow_from_directory('D:\Perkulihan Diki\Semester 5\Pengenalan Pola\Project\dataset', target_size=(64, 64), batch_size=32, class_mode='binary')
    test_set = test_datagen.flow_from_directory('D:\Perkulihan Diki\Semester 5\Pengenalan Pola\Project\dataset', target_size=(64, 64), batch_size=32, class_mode='binary')

    history = model.fit(train_set, steps_per_epoch=len(train_set), epochs=10, validation_data=test_set, validation_steps=len(test_set))

    return history

# Fungsi untuk menyimpan hasil pelatihan dan prediksi ke satu file Excel
def simpan_ke_excel(history, prediksi_daun, filename="hasil_model.xlsx"):
    # Membuat DataFrame dari hasil pelatihan
    df_history = pd.DataFrame({
        'Epoch': range(1, len(history.history['accuracy']) + 1),
        'Accuracy': history.history['accuracy'],
        'Validation Accuracy': history.history['val_accuracy'],
        'Loss': history.history['loss'],
        'Validation Loss': history.history['val_loss']
    })
    
    # Membuat DataFrame untuk hasil prediksi
    df_prediksi = pd.DataFrame(prediksi_daun, columns=["Image Path", "Prediction"])
    
    # Menulis kedua DataFrame ke dalam satu file Excel dengan sheet terpisah
    with pd.ExcelWriter(filename) as writer:
        df_history.to_excel(writer, sheet_name='Hasil Pelatihan', index=False)
        df_prediksi.to_excel(writer, sheet_name='Prediksi', index=False)
    
    print(f"Data hasil pelatihan dan prediksi disimpan dalam {filename}")

# Fungsi untuk memprediksi gambar daun jagung (sehat atau tidak sehat)
def prediksi_daun_jagung(model, image_paths):
    prediksi_daun = []
    
    for image_path in image_paths:
        img = load_img(image_path, target_size=(64, 64))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        result = model.predict(img)
        
        if result[0][0] > 0.5:
            prediction = 'TIDAK SEHAT'
        else:
            prediction = 'SEHAT'
        
        # Menyimpan hasil prediksi ke dalam daftar
        prediksi_daun.append([image_path, prediction])
        
    return prediksi_daun

# Main function
if __name__ == "__main__":
    model = buat_model_cnn()
    history = latih_model_cnn(model)
    
    # Visualisasi hasil pelatihan
    visualisasi_hasil(history)
    
    # Masukkan beberapa path gambar daun jagung
    image_paths = [
        "path_to_image1.jpg",
        "path_to_image2.jpg"
        # Tambahkan path gambar lainnya
    ]
    
    # Prediksi apakah daun jagung sehat atau tidak sehat
    hasil_prediksi = prediksi_daun_jagung(model, image_paths)
    
    # Simpan data hasil pelatihan dan prediksi ke dalam satu file Excel
    simpan_ke_excel(history, hasil_prediksi)
