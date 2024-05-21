import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow import keras
import tensorflow as tf

# Weryfikajcja czy jest GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Limity GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Foldery
my_face_folder = 'D:/PJATK/Semestr_6/WMA---Wizja-maszynowa/Projekt_4/output_frames_me3'
# Skończyło się na użyciu zdjęć
# https://www.kaggle.com/datasets/ashwingupta3012/human-faces?resource=download-directory
youtube_folder = 'D:/PJATK/Semestr_6/WMA---Wizja-maszynowa/Projekt_4/output_frames_kaggle'

# Funkcja do ładowania i przygotowania zdjęć
def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        # Konwersja na skale szarości
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Resize do 200x200
            img = cv2.resize(img, (200, 200))
            # Normalizacja
            img = img.astype('float32') / 255
            images.append(img)
            labels.append(label)
    return images, labels


# Ładowanie danych
my_face, my_label = load_images(my_face_folder, 1)
youtube_face, other_label = load_images(youtube_folder, 0)

# Łączenie zdjęć w jeden set i labelek w jeden set
images = np.array(my_face + youtube_face)
labels = np.array(my_label + other_label)

# Użycie to_categorical
labels = to_categorical(labels, num_classes=2)

# Reshape by pasowało do Conv2D (200px x 200px, szkala szarosci)
images = images.reshape((images.shape[0], 200, 200, 1))

# Split na zbiory testowe i zbiory treningowe
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=1999)

print(x_train.shape)
print(x_test.shape)
print()
print(y_train.shape)
print(y_test.shape)

# Sprawdzenie danych - jak wyglądają
for i in range(20):
    print(y_test[i])
    plt.imshow(x_test[i].reshape(200, 200), cmap='gray')
    plt.show()

# Budowa modelu

# Incicjalizacja
model = Sequential()

# Pierwsza warstwa konwolucyjna - będzie odkrywać mniej dokładnie - np. kształt obiektu
# 32 filtry konwolucyjne (więcej niż 10 bo przekazuje większy obrazek niż w demo)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1)))

# Pierwszy MaxPooling by zredukować wymiary mapy (z 200 do 100), zmiejszyć overfitting (bo mniej cech) itd...
model.add(MaxPooling2D((2, 2)))

# Druga warstwa konwolucyjna - będzie odkrywać bardziej dokładne cechy (nos, oczy itp.)
# 64 filtry konwolucyjne (więcej niż w 1 warstwie bo wchodzimy na większy poziom abstrakcji / skomplikowania)
# Nie podajemy wymiarów, bo keras sobie je po prostu sam weźmie z poprzedniego kroku
model.add(Conv2D(64, (3, 3), activation='relu'))

# Drugi MaxPoling - redukcja do około 50 cech
model.add(MaxPooling2D((2, 2)))

# Spłaszczenie do 1 wymiarowego wektora
model.add(Flatten())

# Pierwsze Dense - więcej dense bo mamy bardziej skomplikowany model
model.add(Dense(64, activation='relu'))

# 2 bo 2 klasy kwalifikatora - ja i ktoś inny
model.add(Dense(2, activation='softmax'))

# Kompilacja modelu
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Zapisanie modelu
model.save('MIW_s24353_f_{}_model_no_fit.h5'.format(1))

# Statystyki modelu
model = keras.models.load_model('MIW_s24353_f_{}_model_no_fit.h5'.format(1))
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('accuracy: {}'.format(acc))
print('loss: {}'.format(loss))

# Wykres loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Wykres accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
