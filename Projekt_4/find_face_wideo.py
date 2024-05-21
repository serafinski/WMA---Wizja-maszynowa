import cv2
import os

img = None
i = 0
object_cascade = None

# Katalog Haarcascade
haarcascade_dir = "C:/Users/tsera/anaconda3/envs/WMA/Library/etc/haarcascades"

# Lista plików Haarcascade
list_haarcascade = ['haarcascade_frontalface_alt2.xml']

# Parametery do detekcji twarzy

# scale factor
f = 105
# min neighbors
n = 5
# min size
s = 200

# Folder do zapisywania zdjec
output_dir = 'output_frames_me3'

# Zapewnienie ze lokalizacja istnieje
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Pozostalosc z orginalnego kodu - ustawienie algorytmu wykrywania
def set_object():
    global i, object_cascade, list_haarcascade
    object_cascade = cv2.CascadeClassifier(os.path.join(haarcascade_dir, list_haarcascade[i]))
    print(list_haarcascade[i])
    i += 1
    if i >= len(list_haarcascade):
        i = 0


def main():
    global img, object_cascade, f, n, s
    video_path = 'D:/PJATK/Semestr_6/WMA---Wizja-maszynowa/Projekt_4/videos/me3.mp4'
    # Inicjalizacja czytania klatek z wideo
    cap = cv2.VideoCapture(video_path)

    # Inicjalizacja algorytmu wykrywania twarzy
    set_object()

    # zmienne do printa
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Czytanie nastepnych klatek
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Przetwarzana klatka: {frame_count + 1}/{total_frames}")

        # Konwersja klatki na skale szarosci
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Wykrywanie twarzy
        faces = object_cascade.detectMultiScale(gray, scaleFactor=f / 100, minNeighbors=n, minSize=(s, s))

        # Zapisywanie klatki z twarza
        if len(faces) > 0:
            # Pierwsza twarz na klatce
            x, y, w, h = faces[0]
            # Wyciecie twarzy i rezize do 200x200
            img = frame[y:y + h, x:x + w]
            img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_LINEAR)
            # Zapisanie twarzy
            cv2.imwrite(os.path.join(output_dir, f'frame_{frame_count:04d}.jpg'), img)

        frame_count += 1

    cap.release()


if __name__ == '__main__':
    main()
