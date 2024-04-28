import cv2
from screeninfo import get_monitors
import os


def upload(i):
    i = i - ord('0')
    image = norm_size(cv2.imread('files/{}'.format(images[i])))
    cv2.imshow('obrazek', image)
    return image


def resize(img, s):
    h, w = img.shape[:2]
    h = h + int(h * s)
    w = w + int(w * s)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

# SKALOWANIE — ROBIŁEM PROJEKT NA EKRANIE 2560 x 1440 - trzeba odpowiednio zeskalowac
def norm_size(img):
    h, w = img.shape[:2]
    if h > 2560 - 400:
        s = (1 - ((1440 - 400) / h)) * (-1)
        img = resize(img, s)
    h, w = img.shape[:2]
    if w > 2560:
        s = (1 - (2560 / w)) * (-1)
        img = resize(img, s)
    return img


# c
def cut():
    global image, image2
    low_color = cv2.getTrackbarPos('low', 'obrazek') * 2
    high_color = cv2.getTrackbarPos('high', 'obrazek') * 2
    ksize = cv2.getTrackbarPos('ksize', 'obrazek') * 2
    if ksize < 10:
        ksize = 10
    image2 = image[low_color:low_color + int((ksize / 2)),
             high_color:high_color + ksize]
    cv2.imshow('obrazek', image2)


# v
def sift1():
    global image
    gimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    siftobject = cv2.SIFT_create()
    keypoint, descriptor = siftobject.detectAndCompute(gimg, None)
    print(descriptor)
    keypointimage = cv2.drawKeypoints(image, keypoint, None, color=(
        0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('obrazek', keypointimage)


# b
def sift2():
    global image
    best_matches_count = 0
    best_matched_img = None

    # Z template'u
    # Obrazek porównywany do referencyjnego
    gimg2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    siftobject = cv2.SIFT_create()
    keypoints_2, descriptors_2 = siftobject.detectAndCompute(gimg2, None)

    # Obrazki referencyjne
    reference_images = os.listdir('data/')
    if not reference_images:
        print("Brak obrazów w folderze 'data/'.")
        return

    # Wysokość i szerokość obrazu, do którego porównujemy
    target_height, target_width = image.shape[:2]

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

        print(f"Liczba dopasowań dla obrazu {img_name}: {len(matches)}")

        # Wybieranie najlepiej dopasowanego obrazka
        if len(matches) > best_matches_count:
            best_matches_count = len(matches)
            matched_img = cv2.drawMatches(
                image2_resized, keypoints_1, image, keypoints_2, matches, None, flags=2)
            best_matched_img = matched_img

    # Wyswietlanie
    if best_matched_img is not None:
        cv2.imshow('obrazek', best_matched_img)
    else:
        print("Nie znaleziono dopasowań.")


# n
def orb():
    global image
    best_matches_count = 0
    best_matched_img = None

    # Z template'u
    # Obrazek porównywany do referencyjnego
    gimg2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints_2, descriptors_2 = orb.detectAndCompute(gimg2, None)

    # Obrazki referencyjne
    reference_images = os.listdir('data/')
    if not reference_images:
        print("Brak obrazów w folderze 'data/'.")
        return

    # Wysokość i szerokość obrazu, do którego porównujemy
    target_height, target_width = image.shape[:2]

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
        keypoints_1, descriptors_1 = orb.detectAndCompute(gimg1, None)

        if descriptors_1 is None or descriptors_2 is None:
            print("Nie można obliczyć deskryptorów dla obrazu.")
            continue

        # Z template'u
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)

        print(f"Liczba dopasowań dla obrazu {img_name}: {len(matches)}")

        # Wybieranie najlepiej dopasowanego obrazka
        if len(matches) > best_matches_count:
            best_matches_count = len(matches)
            matched_img = cv2.drawMatches(
                image2_resized, keypoints_1, image, keypoints_2, matches, None, flags=2)
            best_matched_img = matched_img

    # Wyswietlanie
    if best_matched_img is not None:
        cv2.imshow('obrazek', best_matched_img)
    else:
        print("Nie znaleziono dopasowań.")


def change_h(x):
    global fun
    if fun is not None:
        fun()


images = os.listdir('files/')
image = None
fun = None


def main():
    global image
    global fun
    image = norm_size(cv2.imread('files/{}'.format(images[0])))
    nimg = image.copy()
    cv2.imshow('obrazek', image)
    cv2.createTrackbar('low', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('high', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('ksize', 'obrazek', 5, 255, change_h)
    cv2.createTrackbar('rot', 'obrazek', 0, 360, change_h)

    while True:
        key = cv2.waitKey()
        # -----------wybor obrazka----------------
        if key >= ord('0') and key <= (ord('0') + len(images) - 1):
            image = upload(key)
            nimg = image.copy()
        # ----------------------sift
        elif key == ord('c'):
            cut()
            fun = cut
        elif key == ord('v'):
            sift1()
            fun = sift1
        elif key == ord('b'):
            sift2()
            fun = sift2
        elif key == ord('n'):
            orb()
            fun = orb
        elif key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
