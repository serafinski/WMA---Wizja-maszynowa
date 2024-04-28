import cv2
import numpy as np
import os


def sift(frame):
    best_matches_count = 0
    best_matched_img = None

    # Z template'u
    # Obrazek porównywany do referencyjnego
    gimg2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    siftobject = cv2.SIFT_create()
    keypoints_2, descriptors_2 = siftobject.detectAndCompute(gimg2, None)

    # Obrazki referencyjne
    reference_images = os.listdir('data/')
    if not reference_images:
        print("Brak obrazów w folderze 'data/'.")
        return None

    # Wysokość i szerokość obrazu, do którego porównujemy
    target_height, target_width = frame.shape[:2]

    # Iteracja po obrazkach referencyjnych
    for img_name in reference_images:
        img_path = 'data/{}'.format(img_name)
        image2 = cv2.imread(img_path)

        if image2 is None:
            print(f"Nie udało się załadować obrazu: {img_path}")
            continue

        # Skalowanie obu obrazów do tego samego rozmiaru
        image2_resized = cv2.resize(image2, (target_width, target_height))

        # Z template'u
        gimg1 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)
        keypoints_1, descriptors_1 = siftobject.detectAndCompute(gimg1, None)

        if descriptors_1 is None or descriptors_2 is None:
            print("Nie można obliczyć deskryptorów dla obrazu.")
            continue

        # Z template'u
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Wybieranie najlepiej dopasowanego obrazka
        if len(matches) > best_matches_count:
            best_matches_count = len(matches)
            best_matched_img = cv2.drawMatches(image2_resized, keypoints_1, frame, keypoints_2, matches[:50], None,
                                               flags=2)

    return best_matched_img


# Skrypt do procesowania wideo
video = cv2.VideoCapture('output.mp4')
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = int(video.get(3))
frame_height = int(video.get(4))

# Korekcja wielkości tak by można było miec obrazy side by side
size = (frame_width * 2, frame_height)
result = cv2.VideoWriter(
    'result.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, size)
counter = 1

while True:
    success, frame_rgb = video.read()
    if not success:
        break

    matched_frame = sift(frame_rgb)
    print(f"Przetwarzanie klatki {counter} z {total_frames}")
    counter += 1
    if matched_frame is not None:
        # Pokaż orginalna klatkę i obraz referencyjny z punktami side by side
        result_frame = matched_frame
    else:
        # Jeżeli brak powiązań — pokaż orginalną klatkę
        empty_placeholder = np.zeros_like(frame_rgb)
        result_frame = np.hstack((frame_rgb, empty_placeholder))

    result.write(result_frame)

video.release()
result.release()
