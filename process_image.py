import cv2
import numpy as np


def clean_image(img):

    # (1) Convert to gray, and threshold
    blur = cv2.GaussianBlur(img, (15, 15), 2)
    _, threshed = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY_INV)

    # (2) Close nearby segments + thicken lines
    kernel = np.ones((7, 7))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    return morphed


def show_imgs(*imgs):
    for e, img in enumerate(imgs):
        cv2.imshow(str(e), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img = cv2.imread("imgs/{}.JPG".format(4), 0)
    post_img = clean_image(img)
    show_imgs(post_img)
