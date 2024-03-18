import os
import cv2
import numpy as np


# Wczytuje obraz z dysku, normalizuje jego rozmiar przy użyciu funkcji norm_size i wyświetla go. Argument i to numer
# (w postaci znaku), który jest przekształcany na indeks, aby wybrać obraz z listy dostępnych plików.
def upload(i):
    global files
    # Ekstrakcja numeru obrazka
    i = i - ord('0')
    # Import obrazka i normalizacja rozmiaru
    image = norm_size(cv2.imread('/Users/tomek/Desktop/GIT/WMA---Wizja-maszynowa/13_03_24/images/{}'.format(files[i])))
    # Wyświetlenie obrazka
    cv2.imshow('obrazek', image)
    # Zwrócenie obrazka
    return image


# Funkcja zmieniająca rozmiar obrazka. Argument s to współczynnik skalowania.
def resize(img, s):
    h, w = img.shape[:2]
    h = h + int(h * s)
    w = w + int(w * s)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


# Normalizacja rozmiar obrazu tak, aby jego największy wymiar (wysokość lub szerokość) nie przekraczał 800 pikseli,
# zachowując proporcje obrazu.
def norm_size(img):
    h, w = img.shape[:2]
    if h > w:
        if h > 800:
            s = (1 - (800 / h)) * (-1)
            img = resize(img, s)
    else:
        if w > 800:
            s = (1 - (800 / w)) * (-1)
            img = resize(img, s)
    return img


# Wyodrębnienie określonego zakresu kolorów z obrazu oraz wyświetlenie wynikowej maski.
def hsv_range():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    # Convert the HSV colorspace 
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only blue color 
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    cv2.imshow('obrazek', mask)


def hsv_bitwais():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    # Bitwise-AND mask and original image 
    res = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('obrazek', res)


def hsv_median():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    res = cv2.bitwise_and(image, image, mask=mask)
    res = cv2.medianBlur(res, ksize=ksize)
    cv2.imshow('obrazek', res)


def change_h(x):
    global fun
    if fun is not None:
        fun()


image = None
fun = None
files = None


def main():
    global image, fun, files
    files = os.listdir('/Users/tomek/Desktop/GIT/WMA---Wizja-maszynowa/13_03_24/images/')
    num_image = ord('0')
    image = norm_size(cv2.imread('/Users/tomek/Desktop/GIT/WMA---Wizja-maszynowa/13_03_24/images/{}'.format(files[0])))
    nimg = image.copy()
    cv2.imshow('obrazek', image)
    cv2.createTrackbar('low', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('high', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('ksize', 'obrazek', 5, 50, change_h)

    while True:
        key = cv2.waitKey()
        # -----------Zmiana obrazka----------------
        if ord('0') <= key <= ord('9'):
            num_image = key
            image = upload(key)
            nimg = image.copy()
        # ----------------zmiana rozmiaru---------------
        elif key == ord('-'):
            image = resize(image, -0.1)
            nimg = image.copy()
            cv2.imshow('obrazek', image)
        elif key == ord('+'):
            image = resize(image, 0.1)
            nimg = image.copy()
            cv2.imshow('obrazek', image)
        elif key == ord('='):
            image = upload(num_image)
            cv2.imshow('obrazek', image)
            nimg = image.copy()
        # ----------------kolory------------------------
        # Wyświetlenie obrazka w szarych kolorach
        elif key == ord('q'):
            cv2.imshow('obrazek', cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        # Skala HSV (odcień, nasycenie, wartość)
        elif key == ord('w'):
            nimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            cv2.imshow('obrazek', nimg)
        # Wyodrębnienie określonego zakresu kolorów z obrazu oraz wyświetlenie wynikowej maski.
        elif key == ord('e'):
            hsv_range()
            fun = hsv_range
        # Wyodrębnienie określonego zakresu kolorów z obrazu oraz wyświetlenie wynikowej maski.
        elif key == ord('r'):
            hsv_bitwais()
            fun = hsv_bitwais
        #
        elif key == ord('t'):
            hsv_median()
            fun = hsv_median
        elif key == ord('z'):
            # h = barwa
            cv2.imshow('obrazek', nimg[:, :, 0])
        elif key == ord('x'):
            # s = nasycene
            cv2.imshow('obrazek', nimg[:, :, 1])
        elif key == ord('c'):
            # v = wartość
            cv2.imshow('obrazek', nimg[:, :, 2])
        # ----------------filtry
        elif key == ord('a'):
            cv2.imshow('obrazek', cv2.Canny(image, 55.0, 30.0))
        # Wyświetlenie rozmytego obrazu
        elif key == ord('s'):
            cv2.imshow('obrazek', cv2.blur(image, (7, 7)))
        elif key == ord('d'):
            b = cv2.blur(image, (7, 7))
            cv2.imshow('obrazek', cv2.Canny(b, 55.0, 30.0))
        # Zamkniecie programu
        elif key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
