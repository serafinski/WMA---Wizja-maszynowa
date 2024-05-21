import cv2
import numpy as np
import os
from keras.models import load_model

# Zmienne globalne
object_cascade = None
i = 0

# Ścieżka do katalogu z klasyfikatorami Haar
haarcascade_dir = "/Users/tomek/opt/anaconda3/envs/WMA---Wizja-maszynowa/share/opencv4/haarcascades"

# Lista plików Haarcascade
list_haarcascade = ['haarcascade_frontalface_alt2.xml']

# Parametry do wykrywania twarzy
f = 105
n = 5
s = 200


# Funkcja do ustawiania obiektu klasyfikatora Haar
def set_object():
    global i, object_cascade, list_haarcascade
    object_cascade = cv2.CascadeClassifier(os.path.join(haarcascade_dir, list_haarcascade[i]))
    print(f"Using {list_haarcascade[i]}")
    i += 1
    if i >= len(list_haarcascade):
        i = 0


# Funkcja do przetwarzania obrazu
def preprocess_image(image, face_cascade):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Bierzemy tylko pierwszą wykrytą twarz
        (x, y, w, h) = faces[0]
        face = gray_image[y:y + h, x:x + w]
        resized_face = cv2.resize(face, (200, 200))
        normalized_face = resized_face.astype('float32') / 255.0
        expanded_face = np.expand_dims(normalized_face, axis=(0, -1))
        return expanded_face, (x, y, w, h)
    return None, None


# Główna funkcja
def main():
    global object_cascade, f, n, s

    # Ścieżka do modelu
    # model_path = 'D:/PJATK/Semestr_6/WMA---Wizja-maszynowa/Projekt_4/MIW_s24353_f_1_model_fit.h5'
    model_path = '/Users/tomek/Desktop/GIT/WMA---Wizja-maszynowa/15_05_24/MIW_s24353_f_1_model_no_fit.h5'
    # Załaduj model
    model = load_model(model_path)

    # Ustawienie klasyfikatora Haar
    set_object()

    # Otwarcie kamery
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nie można otworzyć kamery")
        return

    while True:
        ret, frame = cap.read()
        if ret:
            processed_img, face_coords = preprocess_image(frame, object_cascade)

            if processed_img is not None:
                # Przeprowadź predykcję
                prediction = model.predict(processed_img)
                print('Predykcja:', prediction)
                label = 'Ja' if prediction[0][1] > 0.5 else 'Nie ja'
                prob = prediction[0][1] if label == 'Ja' else prediction[0][0]
                print(f"Label: {label}, Probability: {prob:.5f}")

                (x, y, w, h) = face_coords
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f'{label} ({prob:.5f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('Camera', frame)

        key = cv2.waitKey(1)
        if key == 27:  # Naciśnij 'ESC', aby zakończyć
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()