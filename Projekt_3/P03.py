import cv2
from cv2 import bitwise_or
import numpy as np
from screeninfo import get_monitors
import os


def upload(i):
    i = i-ord('0')
    image = norm_size(cv2.imread('pliki/{}'.format(images[i])))
    cv2.imshow('obrazek', image)
    return image


def resize(img, s):
    h, w = img.shape[:2]
    h = h + int(h*s)
    w = w + int(w*s)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def norm_size(img):
    screen = get_monitors()[0]
    h, w = img.shape[:2]
    if h > screen.height-400:
        s = (1 - ((screen.height-400)/h)) * (-1)
        img = resize(img, s)
    h, w = img.shape[:2]
    if w > screen.width:
        s = (1 - (screen.width/w)) * (-1)
        img = resize(img, s)
    return img


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


def morphology():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    kernel = np.ones((4, 4), np.uint8)
    mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('obrazek', mask_without_noise)


def morphology2():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    kernel = np.ones((4, 4), np.uint8)
    mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask_without_noise, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('obrazek', mask_closed)


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


def find_circle():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')

    c_img = image.copy()

    gimg = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)
    bimg = cv2.blur(gimg, (ksize, ksize))
    circles = cv2.HoughCircles(bimg, cv2.HOUGH_GRADIENT, 1.4,
                               low_color, param1=200, param2=60, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(c_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.imshow('obrazek', c_img)


def line():
    global image
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_color, high_color, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 90,
                            minLineLength=100, maxLineGap=5)
    image_l = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_l, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("obrazek", image_l)


def rotate():
    global image
    rot = cv2.getTrackbarPos('rot', 'obrazek')
    height, width = image.shape[:2]
    center_x, center_y = (width/2, height/2)
    M = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)
    rotated_image = cv2.warpAffine(image, M, (width, height))
    cv2.imshow('obrazek', rotated_image)


def cut():
    global image, image2
    low_color = cv2.getTrackbarPos('low', 'obrazek') * 2
    high_color = cv2.getTrackbarPos('high', 'obrazek') * 2
    ksize = cv2.getTrackbarPos('ksize', 'obrazek') * 2
    if ksize < 10:
        ksize = 10
    image2 = image[low_color:low_color+int((ksize/2)),
                   high_color:high_color+ksize]
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
        if key >= ord('0') and key <= (ord('0')+len(images)-1):
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
            cv2.imshow('obrazek', image)
            nimg = image.copy()
    # ----------------kolory------------------------
        elif key == ord('q'):
            cv2.imshow('obrazek', cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        elif key == ord('w'):
            nimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            cv2.imshow('obrazek', nimg)
        elif key == ord('e'):
            hsv_range()
            fun = hsv_range
        elif key == ord('r'):
            hsv_bitwais()
            fun = hsv_bitwais
        elif key == ord('t'):
            hsv_median()
            fun = hsv_median
        elif key == ord('y'):
            # h = barwa
            cv2.imshow('obrazek', nimg[:, :, 0])
        elif key == ord('u'):
            # s = nasycene
            cv2.imshow('obrazek', nimg[:, :, 1])
        elif key == ord('i'):
            # v = wartość
            cv2.imshow('obrazek', nimg[:, :, 2])
    # ----------------filtry
        elif key == ord('a'):
            cv2.imshow('obrazek', cv2.Canny(image, 55.0, 30.0))
        elif key == ord('s'):
            cv2.imshow('obrazek', cv2.blur(image, (7, 7)))
        elif key == ord('d'):
            b = cv2.blur(image, (7, 7))
            cv2.imshow('obrazek', cv2.Canny(b, 55.0, 30.0))
        elif key == ord('f'):
            morphology()
            fun = morphology
        elif key == ord('g'):
            morphology2()
            fun = morphology
        elif key == ord('h'):
            marker()
            fun = marker
        elif key == ord('j'):
            connect_mask()
            fun = connect_mask
        elif key == ord('k'):
            rotate()
            fun = rotate
    # --------------------krztałty
        elif key == ord('z'):
            find_circle()
            fun = find_circle
        elif key == ord('x'):
            line()
            fun = line
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
