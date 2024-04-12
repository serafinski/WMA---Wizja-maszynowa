import cv2
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


# Debug Blur
def gaus_blur_circle():
    ksize = cv2.getTrackbarPos('wiecej', 'obrazek')

    c_img = image.copy()

    gimg = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)
    bimg = cv2.GaussianBlur(gimg, (ksize, ksize), 2)

    cv2.imshow('obrazek', bimg)


# Debug kola
def find_circle():
    # Siła gradientu, by uznać za linie
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    # Siła akumulatora — eliminacja fałszywych detekcji
    high_color = cv2.getTrackbarPos('nowe', 'obrazek')
    # Min dystans między środkami kół — unikniecie kilku kół w tym samym miejscu
    ksize = cv2.getTrackbarPos('wiecej', 'obrazek')

    c_img = image.copy()

    gimg = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)

    # ksize = 3
    bimg = cv2.GaussianBlur(gimg, (3, 3), 2)

    # low_color = param1 - 124
    # high_color = param2 - 48
    # ksize = minDist - 56
    circles = cv2.HoughCircles(bimg, cv2.HOUGH_GRADIENT, 1.4, ksize, param1=low_color, param2=high_color, minRadius=20,
                               maxRadius=40)
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        # Zl
        if i[2] > 32:
            cv2.circle(c_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Gr
        else:
            cv2.circle(c_img, (i[0], i[1]), i[2], (255, 0, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{i[2]}"
        cv2.putText(c_img, text, (i[0], i[1]), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('obrazek', c_img)


# Debug prostokąt - 95
def rectangle():
    low_color = cv2.getTrackbarPos('wiecej', 'obrazek')

    global image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_color, 0, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90, minLineLength=100, maxLineGap=5)
    image_l = image.copy()

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


def rotate():
    global image
    rot = cv2.getTrackbarPos('rot', 'obrazek')
    height, width = image.shape[:2]
    center_x, center_y = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)
    rotated_image = cv2.warpAffine(image, M, (width, height))
    cv2.imshow('obrazek', rotated_image)


def zad1_2():
    ilosc_zlotowek = 0

    ilosc_groszy = 0

    c_img = image.copy()

    gimg = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)
    median_val = np.median(gimg)
    # print(f"Przed: {median_val}")

    # Korekcja kontrastu na tray1.jpg
    if median_val < 52:
        # Kontrola kontrastu
        alpha = 1.1
        gimg = cv2.convertScaleAbs(gimg, alpha=alpha)
        # median_val = np.median(gimg)
        # print(f"Po: {median_val}")

    # 0 B56 R51
    # 1 B56 R52
    # 2 B58 R54
    # 3 B58 R53
    # 4 B59 R55
    # 5 B60 R55
    # 6 B68 R64
    # 7 B57 R53

    bimg = cv2.GaussianBlur(gimg, (3, 3), 2)

    circles = cv2.HoughCircles(bimg, cv2.HOUGH_GRADIENT, 1.4, 56, param1=124, param2=48, minRadius=20,
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
            # cv2.circle(c_img, (i[0], i[1]), 2, (0, 0, 255), 3)

            # Obliczanie i drukowanie powierzchni okręgu
            area = np.pi * (i[2] ** 2)
            print(f"Okrąg o środku ({i[0]}, {i[1]}) ma powierzchnię: {area:.2f} i promień {i[2]:.2f}")

    cv2.imshow('obrazek', c_img)
    print()
    print(f"Ilość 5 złotowek: {ilosc_zlotowek}")
    print(f"Ilość 5 groszówek: {ilosc_groszy}")


def zad3_4():
    print()

    global image

    # Prostokąt — analiza linii
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 95, 0, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90, minLineLength=100, maxLineGap=5)

    # Inicjalizacja wartości do znalezienia najmniejszego/największego x i y
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Aktualizacja min/max x i y
            min_x, min_y = min(min_x, x1, x2), min(min_y, y1, y2)
            max_x, max_y = max(max_x, x1, x2), max(max_y, y1, y2)

    szerokosc = max_x - min_x
    wysokosc = max_y - min_y
    pole_prostokata = szerokosc * wysokosc
    print(f"Pole tacy: {pole_prostokata:.2f}")

    # Okręgi — analiza i zliczanie
    c_img = image.copy()

    gimg = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)
    median_val = np.median(gimg)

    # Korekcja kontrastu na tray1.jpg
    if median_val < 52:
        # Kontrola kontrastu
        alpha = 1.1
        gimg = cv2.convertScaleAbs(gimg, alpha=alpha)
    # gimg = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)

    bimg = cv2.GaussianBlur(gimg, (3, 3), 2)

    circles = cv2.HoughCircles(bimg, cv2.HOUGH_GRADIENT, 1.4, 56, param1=124, param2=48, minRadius=20, maxRadius=40)

    suma_taca = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            area = np.pi * (i[2] ** 2)
            ratio = pole_prostokata / area

            # Złotówki
            if i[2] > 32:
                cv2.circle(c_img, (i[0], i[1]), i[2], (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                zlotowka = f"({i[0]}, {i[1]})Z"
                cv2.putText(c_img, zlotowka, (i[0], i[1]), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                print(f"Złotówka ({i[0]}, {i[1]}) ({i[2]} px promień): {area:.2f}")
                print(f"Złotówka ({i[0]}, {i[1]}) jest mniejsza {ratio:.3f} razy od tacy")

                if min_x <= i[0] <= max_x and min_y <= i[1] <= max_y:
                    suma_taca += 5

            # Groszówki
            else:
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
        if ord('0') <= key <= ord('9'):
            image = upload(key)

        # Zadania
        elif key == ord('q'):
            zad1_2()
            fun = zad1_2
        elif key == ord('w'):
            zad3_4()
            fun = zad3_4

        # Pomocnicze
        elif key == ord('z'):
            find_circle()
            fun = find_circle
        elif key == ord('x'):
            gaus_blur_circle()
            fun = gaus_blur_circle
        elif key == ord('c'):
            rectangle()
            fun = rectangle
        elif key == ord('v'):
            rotate()
            fun = rotate

        # Wyjście
        elif key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
