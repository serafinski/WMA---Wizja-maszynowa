import cv2
import numpy as np

video = cv2.VideoCapture()
video.open('movingball.mp4')
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter(
    'result.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, size)

counter = 1

while True:
    success, frame_rgb = video.read()

    if not success:
        break

    # Konwersja z BGR do HSV
    hsv_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)

    # Czerwona maska
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([255, 255, 255])
    mask_red = cv2.inRange(hsv_frame, lower_red, upper_red)

    # Swiatlo maska
    lower_light = np.array([120, 0, 200])
    upper_light = np.array([255, 255, 255])
    mask_light = cv2.inRange(hsv_frame, lower_light, upper_light)

    # Bitwise-OR - COMBO Czerwony + Swiatlo
    mask = cv2.bitwise_or(mask_red, mask_light)

    # MORPH_OPEN
    kernel = np.ones((15, 15), np.uint8)
    mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # MORPH_CLOSE
    mask_closed = cv2.morphologyEx(mask_without_noise, cv2.MORPH_CLOSE, kernel)

    # Znalezienie konturów
    contours, hierarchy = cv2.findContours(mask_closed, 1, 2)

    # Jezeli jakikolwiek kontur istnieje
    if contours:
        # Najwiekszy kontur na podsawie pola powierzchni
        largest_contour = max(contours, key=cv2.contourArea)
        # Zakladamy, ze najwiekszy kontur to to, co chcemy sledzic
        m = cv2.moments(largest_contour)
        print(m)
        # Sprawdzamy, czy pole powierzchni nie jest rowne 0
        if m['m00'] != 0:
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])

            # Rysowanie markera
            cv2.drawMarker(frame_rgb, (int(cx), int(cy)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
        else:
            print("Brak pola powierzchni")
    else:
        print("Brak konturow")

    result.write(frame_rgb)
    counter = counter + 1

video.release()
result.release()
