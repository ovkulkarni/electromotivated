import cv2
import numpy as np
from itertools import combinations


def clean_image(img):
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, 25, 50)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((7, 7)))
    img = cv2.dilate(img, np.ones((5,5)), iterations=2)
    return img


def detect_lines(img):
    block_size = 6
    aperture = 15
    free_parameter = 0.04

    img = cv2.erode(img, np.ones((9, 9)), iterations=1)

    resps = cv2.cornerHarris(img, block_size, aperture, free_parameter)
    resps = cv2.dilate(resps, None, iterations=4)
    threshold = 0.1

    img = np.copy(img)
    img[resps > threshold * resps.max()] = 0.

    contours,hierarchy = cv2.findContours(img, 1, 2)

    corners = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        p1 = box[0]
        p2 = min(box[1:], key=lambda x: np.linalg.norm(p1-x))

        print(p1, p2)

        im = cv2.drawContours(corners,[box],0,(0,0,255),2)

    return corners


def show_imgs(*imgs):
    for e, img in enumerate(imgs):
        cv2.imshow(str(e), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    for i in range(1, 6):
        img = cv2.imread("imgs/{}.JPG".format(i), 0)
        post_img = clean_image(img)
        line_img = detect_lines(post_img)
        show_imgs(post_img, line_img)
