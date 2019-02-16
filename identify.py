import cv2
import numpy as np

def show_imgs(*imgs):
    for e, img in enumerate(imgs):
        cv2.imshow(str(e), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def identify_component(component_img):
    # check for resistor:
    resistor_img = cv2.erode(component_img, np.ones((9,9)))
    resps = cv2.cornerHarris(resistor_img, 6, 15, 0.04)
    resistor_img *= 0
    resistor_img[resps > .1 * resps.max()] = 255

    resistor_img = cv2.morphologyEx(resistor_img, cv2.MORPH_CLOSE, np.ones((3,3)), iterations=4)
    # resistor_img = cv2.morphologyEx(resistor_img, cv2.MORPH_OPEN, np.ones((9,9)), iterations=1)
    # resistor_img = cv2.morphologyEx(resistor_img, cv2.MORPH_OPEN, np.ones((21,21)), iterations=1)

    # show_imgs(resistor_img, component_img)
    return 'battery'