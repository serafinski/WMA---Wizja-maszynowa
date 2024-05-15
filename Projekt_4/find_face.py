import cv2
import os

fun = None
img = None
i = 0
object_cascade = None

# Haarcascade directory
haarcascade_dir = "C:/Users/tsera/anaconda3/envs/WMA/Library/etc/haarcascades"

# Haarcascade file list
list_haarcascade = ['haarcascade_frontalface_alt2.xml']

# Parameters for face detection
f = 105
n = 5
s = 200

# Output directory for saving frames
#output_dir = 'output_frames'
#output_dir = 'output_frames_720'
output_dir = 'output_yt'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def set_object():
    global i, object_cascade, list_haarcascade
    object_cascade = cv2.CascadeClassifier(os.path.join(haarcascade_dir, list_haarcascade[i]))
    print(list_haarcascade[i])
    i += 1
    if i >= len(list_haarcascade):
        i = 0


def main():
    global fun, img, object_cascade, f, n, s

    # Change video source to a file
    # Wideo YT
    # 1. https://www.youtube.com/watch?v=Xygk7UjKM2g
    # 2. https://www.youtube.com/watch?v=3hvpiK4ttHM
    # 3. https://www.youtube.com/watch?v=-CeLBsqU6qw


    #video_path = 'D:/PJATK/Semestr_6/WMA---Wizja-maszynowa/Projekt_4/videos/WIN_20240515_21_27_46_Pro.mp4'
    #video_path = "D:/PJATK/Semestr_6/WMA---Wizja-maszynowa/Projekt_4/videos/WIN_20240515_21_47_24_Pro.mp4"
    video_path = "D:/PJATK/Semestr_6/WMA---Wizja-maszynowa/Projekt_4/videos/faces.mp4"
    cap = cv2.VideoCapture(video_path)

    set_object()

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame: {frame_count + 1}/{total_frames}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = object_cascade.detectMultiScale(gray, scaleFactor=f / 100, minNeighbors=n, minSize=(s, s))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            img = frame[y:y + h, x:x + w]
            img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(output_dir, f'frame_{frame_count:04d}.jpg'), img)

        frame_count += 1

    cap.release()


if __name__ == '__main__':
    main()
