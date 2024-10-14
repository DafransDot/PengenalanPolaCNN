import tensorflow as tf
import matplotlib.pyplot as plt

# Pastikan path di bawah ini benar
img_path = 'C:\\penpol\\dataset\\train\\class2\\penpol3.jpg'

try:
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [150, 150])
    img = img / 255.0

    # Ubah tensor menjadi numpy array untuk matplotlib
    img = img.numpy() 

    plt.imshow(img)
    plt.axis('off')  
    plt.show()
except tf.errors.NotFoundError:
    print(f"File tidak ditemukan: {img_path}")