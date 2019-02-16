import cv2
import numpy as np


def show_imgs(*imgs):
    for e, img in enumerate(imgs):
        cv2.imshow(str(e), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def identify_component(component_img, is_horizontal):
    # check for voltimeter/ammeter circles:
    circles_img = cv2.erode(component_img, np.ones((9, 9)))
    circles_img = cv2.morphologyEx(
        circles_img, cv2.MORPH_CLOSE, np.ones((5, 5)), iterations=5)
    circles_img = cv2.morphologyEx(
        circles_img, cv2.MORPH_OPEN, np.ones((20, 20)), iterations=1)

    contours, _ = cv2.findContours(circles_img, 1, 2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        arclen = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (arclen * arclen)
        if (circularity > .8):
            return 'voltimeter/ammeter'

    # check for resistor:
    color_img = cv2.cvtColor(component_img, cv2.COLOR_GRAY2BGR)

    resistor_img = cv2.erode(component_img, np.ones((9, 9)))

    contours, _ = cv2.findContours(resistor_img, 1, 2)
    if len(contours) == 1:
        cnt = contours[0]
        epsilon = 0.01*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        count = 0
        if is_horizontal:
            approx = sorted(approx, key=lambda x: x[0][0])
            going_up = approx[0][0][1] - approx[1][0][1] > 0
            for p1, p2 in zip(approx, approx[1:]):
                cv2.line(color_img, tuple(np.int0(p1)[0]),
                         tuple(np.int0(p2)[0]), (255, 0, 0), 2)
                if going_up != (p2[0][1] - p1[0][1] > 0):
                    going_up = not going_up
                    count += 1
        else:
            approx = sorted(approx, key=lambda x: x[0][1])
            going_up = approx[0][0][0] - approx[1][0][0] > 0
            for p1, p2 in zip(approx, approx[1:]):
                cv2.line(color_img, tuple(np.int0(p1)[0]),
                         tuple(np.int0(p2)[0]), (255, 0, 0), 2)
                if going_up != (p2[0][0] - p1[0][0] > 0):
                    going_up = not going_up
                    count += 1
        if count > 4:
            return 'resistor/inductor'

    # check for battery/capacitor
    if len(contours) > 1:
        kernel = np.ones((30, 1)) if is_horizontal else np.ones((1, 30))
        relevant_img = cv2.morphologyEx(component_img, cv2.MORPH_OPEN, kernel)
        real_contours, _ = cv2.findContours(relevant_img, 1, 2)
        if len(real_contours):
            largest = max(max(w, h) for x, y, w, h in [cv2.boundingRect(cnt)
                                                       for cnt in real_contours])
            smallest = min(max(w, h) for x, y, w, h in [cv2.boundingRect(cnt)
                                                        for cnt in real_contours])
            return 'battery' if largest / smallest > 1.5 else 'capacitor'
    return 'evan is gay'
