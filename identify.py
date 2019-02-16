import cv2
import numpy as np

def show_imgs(*imgs):
    for e, img in enumerate(imgs):
        cv2.imshow(str(e), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def identify_component(component_img):
    # check for voltimeter/ammeter circles:
    circles_img = cv2.erode(component_img, np.ones((9,9)))
    circles_img = cv2.morphologyEx(circles_img, cv2.MORPH_CLOSE, np.ones((5,5)), iterations=5)
    circles_img = cv2.morphologyEx(circles_img, cv2.MORPH_OPEN, np.ones((20,20)), iterations=1)

    contours, _ = cv2.findContours(circles_img, 1, 2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        arclen = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (arclen * arclen)
        if (circularity > .8):
            return 'voltimeter/ammeter'

    # check for resistor:
    color_img = cv2.cvtColor(component_img, cv2.COLOR_GRAY2BGR)

    resistor_img = cv2.erode(component_img, np.ones((9,9)))
    # resps = cv2.cornerHarris(resistor_img, 6, 15, 0.04)
    # resistor_img *= 0
    # resistor_img[resps > .1 * resps.max()] = 255

    contours, _ = cv2.findContours(resistor_img, 1, 2)
    for cnt in contours:
        epsilon = 0.01*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        print(approx)

    # resistor_img = cv2.morphologyEx(resistor_img, cv2.MORPH_CLOSE, np.ones((3,3)), iterations=4)
    # resistor_img = cv2.morphologyEx(resistor_img, cv2.MORPH_OPEN, np.ones((9,9)), iterations=1)
    # resistor_img = cv2.morphologyEx(resistor_img, cv2.MORPH_OPEN, np.ones((21,21)), iterations=1)

    show_imgs(color_img)
    return 'battery'