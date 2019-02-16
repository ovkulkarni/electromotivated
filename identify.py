import cv2
import numpy as np

def show_imgs(*imgs):
    for e, img in enumerate(imgs):
        cv2.imshow(str(e), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def identify_component(component_img):
    show_imgs(component_img)
    return 'battery'