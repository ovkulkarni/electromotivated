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


def detect_graph_components(img):
    block_size = 6
    aperture = 15
    free_parameter = 0.04

    img = cv2.erode(img, np.ones((9, 9)), iterations=1)

    resps = cv2.cornerHarris(img, block_size, aperture, free_parameter)
    resps = cv2.dilate(resps, None, iterations=8)
    threshold = 0.1

    img = np.copy(img)
    img[resps > threshold * resps.max()] = 0.

    corner_img = img*0
    corner_img[resps > threshold * resps.max()] = 255
    corner_contours, _ = cv2.findContours(corner_img, 1, 2)
    corners = []
    for cnt in corner_contours:
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01'] / M['m00'])
        corners.append((cx, cy))

    contours,hierarchy = cv2.findContours(img, 1, 2)
    line_segments = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        p1 = box[0]
        p2 = min(box[1:], key=lambda x: np.linalg.norm(p1-x))

        p3, p4 = [p for p in box if not np.array_equal(p, p1) and not np.array_equal(p, p2)]

        line_segments.append((np.int0((p1 + p2) / 2), np.int0((p3 + p4) / 2)))

    return corners, line_segments


def show_imgs(*imgs):
    for e, img in enumerate(imgs):
        cv2.imshow(str(e), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    for i in range(1, 6):
        img = cv2.imread("imgs/{}.JPG".format(i), 0)
        post_img = clean_image(img)
        corners, line_segments = detect_graph_components(post_img)

        line_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for line in line_segments:
            cv2.line(line_img, tuple(line[0]), tuple(line[1]), (0,0,255), 1)

        for corner in corners:
            cv2.circle(line_img, corner, 1, (255,0,0), -1)

        show_imgs(img, line_img)
