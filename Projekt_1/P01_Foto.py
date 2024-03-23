import cv2
import numpy as np


def norm_size(img):
    h, w = img.shape[:2]
    if h > w:
        if h > 800:
            img = cv2.resize(img, (int(w * 800 / h), 800), interpolation=cv2.INTER_LINEAR)
    else:
        if w > 800:
            img = cv2.resize(img, (800, int(h * 800 / w)), interpolation=cv2.INTER_LINEAR)
    return img


# Zmień format obrazu na HSV
def hsv_range():
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Utwórz maskę kolorów, jakie znajdują się na piłce
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([255, 255, 255])
    mask_red = cv2.inRange(hsv_frame, lower_red, upper_red)

    lower_light = np.array([120, 0, 200])
    upper_light = np.array([255, 255, 255])
    mask_light = cv2.inRange(hsv_frame, lower_light, upper_light)

    # Przy pomocy operacji binarnej
    mask = cv2.bitwise_or(mask_red, mask_light)
    cv2.imshow('obrazek', mask)


# Popraw jakość obrazu (usuń szum)
def morphology():
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([255, 255, 255])
    mask_red = cv2.inRange(hsv_frame, lower_red, upper_red)

    lower_light = np.array([120, 0, 200])
    upper_light = np.array([255, 255, 255])
    mask_light = cv2.inRange(hsv_frame, lower_light, upper_light)

    mask = cv2.bitwise_or(mask_red, mask_light)

    # Poprzez operacje morfologiczne
    kernel = np.ones((15, 15), np.uint8)
    mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask_without_noise, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('obrazek', mask_closed)


# Oblicz środek obiektu i dodaj marker do obrazu oznaczający środek obiektu.
def marker():
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([255, 255, 255])
    mask_red = cv2.inRange(hsv_frame, lower_red, upper_red)

    lower_light = np.array([120, 0, 200])
    upper_light = np.array([255, 255, 255])
    mask_light = cv2.inRange(hsv_frame, lower_light, upper_light)

    mask = cv2.bitwise_or(mask_red, mask_light)

    kernel = np.ones((15, 15), np.uint8)
    mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask_without_noise, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(mask_closed, 1, 2)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        m = cv2.moments(largest_contour)
        print(m)
        if m['m00'] != 0:
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            image_marker = image.copy()
            cv2.drawMarker(image_marker, (int(cx), int(cy)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS,
                           thickness=2)
        else:
            print("Brak pola powierzchni")
    else:
        print("Brak konturow")
    cv2.imshow('obrazek', image_marker)


def change_h(x):
    global fun
    if fun is not None:
        fun()


image = None
fun = None
files = None


def main():
    global image, fun, files
    image_path = '/Users/tomek/Desktop/GIT/WMA---Wizja-maszynowa/Projekt_1/Assets/ball.png'
    image = norm_size(cv2.imread(image_path))
    cv2.imshow('obrazek', image)
    cv2.createTrackbar('low', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('high', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('ksize', 'obrazek', 5, 50, change_h)

    while True:
        key = cv2.waitKey()
        if key == ord('q'):
            hsv_range()
            fun = hsv_range
        elif key == ord('w'):
            morphology()
            fun = morphology
        elif key == ord('e'):
            marker()
            fun = marker
        elif key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
