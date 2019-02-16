import cv2
import numpy as np


def clean_image(img):

    # (1) Convert to gray, and threshold

    _, threshed = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)

    # (2) Morph-op to remove noise
    kernel = np.ones((7, 7))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((2, 2))
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
    return morphed


def show_imgs(*imgs):
    for e, img in enumerate(imgs):
        cv2.imshow(str(e), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    for i in range(1, 6):
        img = cv2.imread("imgs/{}.JPG".format(i), 0)
        post_img = clean_image(img)
        show_imgs(post_img)
