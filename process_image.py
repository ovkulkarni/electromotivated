import cv2
import numpy as np


def clean_image(img):
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, 25, 50)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((7, 7)))
    return img


def detect_lines(img):
    low_threshold = 50
    high_threshold = 150
    img = cv2.Canny(img, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid

    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    threshold = 15
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_img = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 5)

    return img


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
