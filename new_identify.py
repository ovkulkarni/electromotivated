import cv2
import numpy as np


def show_imgs(*imgs):
    for e, img in enumerate(imgs):
        cv2.imshow(str(e), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def identify_component(component_img, circles_component, sans_dilate, is_horizontal):
    # check for voltimeter/ammeter circles:
    contours, _ = cv2.findContours(circles_component, 1, 2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        arclen = cv2.arcLength(cnt, True)
        if arclen == 0:
            continue
        circularity = (4 * np.pi * area) / (arclen * arclen)
        if (circularity > .8):
            return 'voltmeter'

    # check for diode

    # check for resistor:
    resistor_img = cv2.morphologyEx(
        component_img, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=2)
    resistor_img = cv2.erode(resistor_img, np.ones((7, 7)))
    # color_img = cv2.cvtColor(resistor_img, cv2.COLOR_GRAY2BGR)

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
                if going_up != (p2[0][1] - p1[0][1] > 0):
                    going_up = not going_up
                    count += 1
        else:
            approx = sorted(approx, key=lambda x: x[0][1])
            going_up = approx[0][0][0] - approx[1][0][0] > 0
            for p1, p2 in zip(approx, approx[1:]):
                if going_up != (p2[0][0] - p1[0][0] > 0):
                    going_up = not going_up
                    count += 1
        if count > 4:
            # check for inductor
            resistor_img = cv2.morphologyEx(sans_dilate, cv2.MORPH_OPEN,
                                        np.ones((2, 15) if is_horizontal else (15, 2)))
            contours, _ = cv2.findContours(resistor_img, 1, 2)
            if len(contours) > 2:
                return 'inductor'
            return 'resistor'

    # # checking for switch
    switch_img = cv2.morphologyEx(component_img, cv2.MORPH_CLOSE, np.ones((3,3)))
    contours, _ = cv2.findContours(switch_img, 1, 2)
    if (len(contours) == 2):
        c1 = cv2.contourArea(contours[0])
        c2 = cv2.contourArea(contours[1])
        ratio = (c1/c2) if c1 > c2 else (c2/c1)
        if ratio > 4:
            return 'switch'
        elif ratio > 2:
            switch_img = component_img
            sym_img = cv2.flip(component_img, 0 if is_horizontal else 1)
            sym_score = cv2.countNonZero(
                switch_img & sym_img) / cv2.countNonZero(switch_img)
            if sym_score < 0.5:
                return 'switch'

    # check for battery/capacitor
    contours, _ = cv2.findContours(component_img, 1, 2)
    if len(contours) > 1:
        kernel = np.ones((30, 1)) if is_horizontal else np.ones((1, 30))
        relevant_img = cv2.morphologyEx(component_img, cv2.MORPH_OPEN, kernel)
        real_contours, _ = cv2.findContours(relevant_img, 1, 2)
        if len(real_contours):
            largest = max((cv2.boundingRect(cnt) for cnt in real_contours),
                          key=lambda x: max(x[2], x[3]))
            smallest = min((cv2.boundingRect(cnt) for cnt in real_contours),
                           key=lambda x: max(x[2], x[3]))
            if max(largest[2], largest[3]) / max(smallest[2], smallest[3]) > 1.5:
                if is_horizontal:
                    return '{}battery'.format('left' if largest[0] < smallest[0] else 'right')
                else:
                    return '{}battery'.format('top' if largest[1] > smallest[1] else 'bottom')
            else:
                return 'capacitor'
    return 'evan is gay'
