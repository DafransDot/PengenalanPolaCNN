import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

    train_set = train_datagen.flow_from_directory('dataset/', target_size=(64, 64), batch_size=32, class_mode='binary')
    test_set = test_datagen.flow_from_directory('dataset/', target_size=(64, 64), batch_size=32, class_mode='binary')

    history = model.fit(train_set, steps_per_epoch=len(train_set), epochs=10, validation_data=test_set, validation_steps=len(test_set))

    return history

# Visualisasi hasil pelatihan
def visualisasi_hasil(history):
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

# Fungsi untuk memprediksi gambar daun jagung (sehat atau tidak sehat)
def prediksi_daun_jagung(model, image_path):
    img = load_img(image_path, target_size=(64, 64))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    result = model.predict(img)
    
    if result[0][0] > 0.5:
        print(f'Gambar {image_path} adalah daun jagung TIDAK SEHAT')
    else:
        print(f'Gambar {image_path} adalah daun jagung SEHAT')

# Main function
if __name__ == "__main__":
    model = buat_model_cnn()
    history = latih_model_cnn(model)
    
    # Visualisasi hasil pelatihan
    visualisasi_hasil(history)
    
    # Meminta input gambar dari pengguna
    image_path = input("Masukkan path gambar daun jagung: ")
    
    # Prediksi apakah daun jagung sehat atau tidak sehat
    prediksi_daun_jagung(model, image_path)
