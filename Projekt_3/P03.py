import cv2
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


def sift1():
    global image
    gimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    siftobject = cv2.SIFT_create()
    keypoint, descriptor = siftobject.detectAndCompute(gimg, None)
    print(descriptor)
    keypointimage = cv2.drawKeypoints(image, keypoint, None, color=(
        0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('obrazek', keypointimage)


def sift2():
    global image, image2
    gimg1 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gimg2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    siftobject = cv2.SIFT_create()
    keypoints_1, descriptors_1 = siftobject.detectAndCompute(gimg1, None)
    keypoints_2, descriptors_2 = siftobject.detectAndCompute(gimg2, None)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    matched_img = cv2.drawMatches(
        image2, keypoints_1, image, keypoints_2, matches, image, flags=2)
    cv2.imshow('obrazek', matched_img)


def sift3():
    global image, image2
    k = cv2.getTrackbarPos('ksize', 'obrazek')
    gimg1 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    if (k % 2) != 1:
        k += 1
    gimg1 = cv2.medianBlur(gimg1, ksize=k)
    gimg2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gimg2 = cv2.medianBlur(gimg2, ksize=k)
    siftobject = cv2.SIFT_create()
    keypoints_1, descriptors_1 = siftobject.detectAndCompute(gimg1, None)
    keypoints_2, descriptors_2 = siftobject.detectAndCompute(gimg2, None)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)

    matches = sorted(matches, key=lambda x: x.distance)
    matched_img = cv2.drawMatches(
        image2, keypoints_1, image, keypoints_2, matches, image, flags=2)
    cv2.imshow('obrazek', matched_img)


def orb():
    global image, image2
    gimg1 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gimg2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints_1, descriptors_1 = orb.detectAndCompute(gimg1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(gimg2, None)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    matched_img = cv2.drawMatches(
        image2, keypoints_1, image, keypoints_2, matches, image, flags=2)
    cv2.imshow('obrazek', matched_img)


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
            sift3()
            fun = sift3

        elif key == ord('m'):
            orb()
            fun = orb
        elif key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
