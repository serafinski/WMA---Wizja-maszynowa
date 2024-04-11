import cv2
from cv2 import bitwise_or
import numpy as np
from screeninfo import get_monitors
import os


def upload(i):
    i = i - ord('0')
    image = norm_size(cv2.imread('pliki/{}'.format(images[i])))
    cv2.imshow('obrazek', image)
    return image


def resize(img, s):
    h, w = img.shape[:2]
    h = h + int(h * s)
    w = w + int(w * s)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def norm_size(img):
    screen = get_monitors()[0]
    h, w = img.shape[:2]
    if h > screen.height - 400:
        s = (1 - ((screen.height - 400) / h)) * (-1)
        img = resize(img, s)
    h, w = img.shape[:2]
    if w > screen.width:
        s = (1 - (screen.width / w)) * (-1)
        img = resize(img, s)
    return img


def marker():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')

    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])

    mask = cv2.inRange(hsv_frame, lower, upper)
    contours, hierarchy = cv2.findContours(mask, 1, 2)
    M = cv2.moments(contours[0])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    image_marker = image.copy()
    cv2.drawMarker(image_marker, (int(cx), int(cy)), color=(
        0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
    cv2.imshow('obrazek', image_marker)


def connect_mask():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')

    # Convert the HSV colorspace
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only blue color
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    lower = np.array([0, 100, 100])
    upper = np.array([ksize, 255, 255])
    mask2 = cv2.inRange(hsv_frame, lower, upper)
    b_mask = bitwise_or(mask, mask2)
    cv2.imshow('obrazek', b_mask)


def blur_circle():
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')

    c_img = image.copy()

    gimg = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)
    bimg = cv2.blur(gimg, (ksize, ksize))

    cv2.imshow('obrazek', bimg)


def find_circle():
    #gradient by linia
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    #sila falszywe dektekcje
    high_color = cv2.getTrackbarPos('nowe', 'obrazek')
    #min dystans miedzy kolami
    ksize = cv2.getTrackbarPos('wiecej', 'obrazek')

    c_img = image.copy()

    gimg = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)

    bimg = cv2.blur(gimg, (3, 3))

    # 0, 1, 2, 4, 7
    # 3, 6
    # 5

    # low_color = param1 162 - 28 / dla 30/40 - 101
    # high_color = param2 60 - 48 / dla 30/40 - 48
    # ksize = minDist 13-63 - 53 / dla 30/40 56
    # ksize = 3
    circles = cv2.HoughCircles(bimg, cv2.HOUGH_GRADIENT, 1.4, ksize, param1=low_color, param2=high_color, minRadius=20,
                               maxRadius=40)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        if i[2] > 32:
            cv2.circle(c_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        else:
            cv2.circle(c_img, (i[0], i[1]), i[2], (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{i[2]}"
        cv2.putText(c_img, text, (i[0], i[1]), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('obrazek', c_img)


def zad1_2():
    ilosc_zlotowek = 0

    ilosc_groszy = 0

    c_img = image.copy()

    gimg = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)
    bimg = cv2.blur(gimg, (3, 3))

    circles = cv2.HoughCircles(bimg, cv2.HOUGH_GRADIENT, 1.4, 56, param1=101, param2=48, minRadius=20,
                               maxRadius=40)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print()
        for i in circles[0, :]:
            # Rysowanie okręgu
            if i[2] > 32:
                ilosc_zlotowek += 1
                cv2.circle(c_img, (i[0], i[1]), i[2], (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                zlotowka = "Z"
                cv2.putText(c_img, zlotowka, (i[0], i[1]), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                ilosc_groszy += 1
                cv2.circle(c_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                zlotowka = "G"
                cv2.putText(c_img, zlotowka, (i[0], i[1]), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            # Rysowanie środka okręgu
            #cv2.circle(c_img, (i[0], i[1]), 2, (0, 0, 255), 3)

            # Obliczanie i drukowanie powierzchni okręgu
            area = np.pi * (i[2] ** 2)
            print(f"Okrąg o środku ({i[0]}, {i[1]}) ma powierzchnię: {area:.2f} i promien {i[2]:.2f}")

    cv2.imshow('obrazek', c_img)
    print()
    print(f"Ilosc 5 zlotowek: {ilosc_zlotowek}")
    print(f"Ilosc 5 groszowek: {ilosc_groszy}")


# 95
def rectangle():
    global image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 95, 0, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90, minLineLength=100, maxLineGap=5)
    image_l = image.copy()

    # Inicjalizacja wartości do znalezienia najmniejszego/największego x i y
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Aktualizacja min/max x i y
        min_x, min_y = min(min_x, x1, x2), min(min_y, y1, y2)
        max_x, max_y = max(max_x, x1, x2), max(max_y, y1, y2)

        cv2.line(image_l, (x1, y1), (x2, y2), (0, 255, 0), 2)

    szerokosc = max_x - min_x
    wysokosc = max_y - min_y
    pole = szerokosc * wysokosc

    # Rysowanie prostokąta na podstawie znalezionych współrzędnych
    cv2.rectangle(image_l, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
    print(f"Szerokosc: {szerokosc}, Wysokosc: {wysokosc},Pole: {pole}")

    cv2.imshow("obrazek", image_l)


def zad3_4():
    print()
    global image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Prostokąt - analiza linii
    edges = cv2.Canny(gray, 95, 0, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90, minLineLength=100, maxLineGap=5)
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            min_x, min_y = min(min_x, x1, x2), min(min_y, y1, y2)
            max_x, max_y = max(max_x, x1, x2), max(max_y, y1, y2)

    szerokosc = max_x - min_x
    wysokosc = max_y - min_y
    pole_prostokata = szerokosc * wysokosc
    print(f"Pole tacy: {pole_prostokata:.2f}")

    # Okręgi - analiza i zliczanie
    c_img = image.copy()
    gimg = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)
    bimg = cv2.blur(gimg, (3, 3))
    circles = cv2.HoughCircles(bimg, cv2.HOUGH_GRADIENT, 1.4, 56, param1=101, param2=48, minRadius=20, maxRadius=40)

    suma_taca = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            area = np.pi * (i[2] ** 2)
            ratio = pole_prostokata / area
            if i[2] > 32:  # Złotówki
                cv2.circle(c_img, (i[0], i[1]), i[2], (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                zlotowka = f"({i[0]}, {i[1]})Z"
                cv2.putText(c_img, zlotowka, (i[0], i[1]), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                print(f"Złotówka ({i[0]}, {i[1]}) ({i[2]} px promień): {area:.2f}")
                print(f"Złotówka ({i[0]}, {i[1]}) jest mniejsza {ratio:.3f} razy od tacy")
                if min_x <= i[0] <= max_x and min_y <= i[1] <= max_y:
                    suma_taca += 5
            else:  # Groszówki
                cv2.circle(c_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                zlotowka = "G"
                cv2.putText(c_img, zlotowka, (i[0], i[1]), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

                if min_x <= i[0] <= max_x and min_y <= i[1] <= max_y:
                    suma_taca += 0.05

    # Rysowanie prostokąta na podstawie znalezionych współrzędnych
    cv2.rectangle(c_img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
    print()
    print(f"Suma taca: {suma_taca:.2f} zl")
    cv2.imshow("obrazek", c_img)


def rotate():
    global image
    rot = cv2.getTrackbarPos('rot', 'obrazek')
    height, width = image.shape[:2]
    center_x, center_y = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)
    rotated_image = cv2.warpAffine(image, M, (width, height))
    cv2.imshow('obrazek', rotated_image)


def change_h(x):
    global fun
    if fun is not None:
        fun()


images = os.listdir('pliki/')
image = None
fun = None


def main():
    global image
    global fun
    image = norm_size(cv2.imread('pliki/{}'.format(images[0])))
    cv2.imshow('obrazek', image)
    cv2.createTrackbar('low', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('nowe', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('wiecej', 'obrazek', 5, 255, change_h)
    cv2.createTrackbar('rot', 'obrazek', 0, 360, change_h)

    while True:
        key = cv2.waitKey()
        # -----------wybor obrazka----------------
        if key >= ord('0') and key <= ord('9'):
            image = upload(key)

        # Zadania
        elif key == ord('q'):
            zad1_2()
            fun = zad1_2
        elif key == ord('w'):
            zad3_4()
            fun = zad3_4
        # Pomocnicze
        elif key == ord('e'):
            find_circle()
            fun = find_circle
        elif key == ord('r'):
            blur_circle()
            fun = blur_circle
        elif key == ord('t'):
            rectangle()
            fun = rectangle
        elif key == ord('a'):
            marker()
            fun = marker
        elif key == ord('s'):
            connect_mask()
            fun = connect_mask
        elif key == ord('d'):
            rotate()
            fun = rotate
        # Wyjscie
        elif key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
