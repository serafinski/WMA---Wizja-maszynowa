import cv2
import numpy as np
import os

def find_best_match(frame):
    best_matches_count = 0
    best_matched_img = None
    gimg2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints_2, descriptors_2 = orb.detectAndCompute(gimg2, None)

    reference_images = os.listdir('data/')
    if not reference_images:
        print("Brak obrazów w folderze 'data/'.")
        return None

    target_height, target_width = frame.shape[:2]

    for img_name in reference_images:
        img_path = 'data/{}'.format(img_name)
        image2 = cv2.imread(img_path)
        if image2 is None:
            print(f"Nie udało się załadować obrazu: {img_path}")
            continue

        image2_resized = cv2.resize(image2, (target_width, target_height))
        gimg1 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)
        keypoints_1, descriptors_1 = orb.detectAndCompute(gimg1, None)

        if descriptors_1 is None or descriptors_2 is None:
            print("Nie można obliczyć deskryptorów dla obrazu.")
            continue

        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > best_matches_count:
            best_matches_count = len(matches)
            best_matched_img = cv2.drawMatches(image2_resized, keypoints_1, frame, keypoints_2, matches[:50], None, flags=2)

    return best_matched_img

# Original video processing script
video = cv2.VideoCapture('output.mp4')
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width * 2, frame_height)  # Adjust size to accommodate two images side by side
result = cv2.VideoWriter(
    'result_orb.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, size)
counter = 1

while True:
    success, frame_rgb = video.read()
    if not success:
        break

    matched_frame = find_best_match(frame_rgb)
    print(f"Przetwarzanie klatki {counter} z {total_frames}")
    counter += 1
    if matched_frame is not None:
        # Display the original and matched images with keypoints side by side
        result_frame = matched_frame
    else:
        # No matches found, just show the original frame
        empty_placeholder = np.zeros_like(frame_rgb)
        result_frame = np.hstack((frame_rgb, empty_placeholder))

    result.write(result_frame)

video.release()
result.release()
