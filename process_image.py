import cv2
import numpy as np
from collections import deque, namedtuple
from unionfind import UnionFind
from new_identify import identify_component
from graph import build_graph, component_edges, build_circuit

PLUSPLUSPLUS = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
]).astype(np.uint8)

def resize_image(img):
    scale = np.sqrt(4e5 / (img.shape[0] * img.shape[1]))
    new_bounds = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    img = cv2.resize(img, new_bounds, interpolation=cv2.INTER_CUBIC)
    return img


def clean_image(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 25, 30)
    img2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((7, 7)))
    img2 = cv2.dilate(img2, np.ones((5, 5)), iterations=2)
    return img2, img


def detect_graph_components(img):
    # detect the circles
    circles_img = cv2.morphologyEx(
        img, cv2.MORPH_CLOSE, np.ones((5, 5)), iterations=3)

    contours, _ = cv2.findContours(255-circles_img, 1, 2)
    for cnt in contours:
        if cv2.contourArea(cnt) < 5000:
            cv2.drawContours(circles_img, [cnt], -1, 255, -1)
    circles_img = cv2.erode(circles_img, np.ones((21, 21)))

    contours, _ = cv2.findContours(circles_img, 1, 2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        arclen = cv2.arcLength(cnt, True)
        if arclen == 0:
            continue
        circularity = (4 * np.pi * area) / (arclen * arclen)
        if (circularity < .8):
            cv2.drawContours(circles_img, [cnt], -1, 0, -1)
    circles_img = cv2.dilate(circles_img, np.ones((21, 21)))

    # bind together
    blob_img = img-circles_img
    blob_img = cv2.morphologyEx(blob_img, cv2.MORPH_CLOSE, np.ones((29,29)), iterations=1)
    blob_img = cv2.erode(blob_img, np.ones((9, 9)), iterations=2)
    blob_img = cv2.morphologyEx(blob_img, cv2.MORPH_CLOSE, np.ones((15,15)), iterations=1)
    blob_img = cv2.morphologyEx(blob_img, cv2.MORPH_OPEN, np.ones((10,10)), iterations=1)
    blob_img = cv2.dilate(blob_img, np.ones((29,29)))

    obstacle_mask = 255 - (circles_img | blob_img)
    line_img = cv2.erode(img&obstacle_mask, np.ones((9,9)))

    resps = cv2.cornerHarris(line_img, 6, 15, 0.04)
    corner_img = img*0
    corner_img[resps > .01 * resps.max()] = 255

    corner_img = cv2.morphologyEx(corner_img, cv2.MORPH_CLOSE, np.ones((15,15)))

    corner_contours, _ = cv2.findContours(corner_img, 1, 2)
    corners = []
    for cnt in corner_contours:
        if cv2.contourArea(cnt) < 500:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01'] / M['m00'])
                corners.append((cx, cy))
        else:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

    corner_img *=0
    for c in corners:
        corner_img[c[1]][c[0]] = 255
    corner_img = cv2.dilate(corner_img, PLUSPLUSPLUS)

    line_img -= corner_img
    line_img[line_img < 10] = 0

    # find lines
    contours, _ = cv2.findContours(line_img, 1, 2)
    line_segments = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        p1 = box[0]
        p2 = min(box[1:], key=lambda x: np.linalg.norm(p1-x))

        p3, p4 = [p for p in box if not np.array_equal(
            p, p1) and not np.array_equal(p, p2)]

        ratio = np.linalg.norm(p1-p2) / np.linalg.norm(p2-p3)
        if ratio < 1: ratio = 1/ratio

        if ratio < 3:
            continue

        line_segments.append((np.int0((p1 + p2) / 2), np.int0((p3 + p4) / 2)))

    return corners, line_segments


def classify_components(img, cedges, graph, post_sans_dilate_img):
    locations = [(tuple(graph[line[0]].loc), tuple(graph[line[1]].loc))
                 for line in cedges]

    original_img = np.copy(img)

    # stretch the lengths by 20%
    locs = []  # sorry :(
    midpoints = []
    orientations = {}
    for loc in locations:
        a, b = map(np.array, loc)
        midpoint = (a + b) / 2
        scale = (np.linalg.norm(a-midpoint) + 30) / np.linalg.norm(a-midpoint)
        p1 = ((a - midpoint)*scale) + midpoint
        p2 = midpoint - ((a - midpoint)*scale)
        locs.append((tuple(np.int0(p1)), tuple(np.int0(p2))))
        midpoints.append(tuple(np.int0(midpoint)))
        orientations[tuple(np.int0(midpoint))] = abs(
            a[0] - b[0]) > abs(a[1] - b[1])

    img = np.copy(img)
    for loc in locs:  # connect the contours completely
        cv2.line(img, loc[0], loc[1], 255, 10)
        cv2.circle(img, loc[0], 15, 0, -1)
        cv2.circle(img, loc[1], 15, 0, -1)

    # find the circles:
    circles_img = cv2.morphologyEx(
        original_img, cv2.MORPH_CLOSE, np.ones((5, 5)), iterations=3)

    contours, _ = cv2.findContours(255-circles_img, 1, 2)
    for cnt in contours:
        if cv2.contourArea(cnt) < 5000:
            cv2.drawContours(circles_img, [cnt], -1, 255, -1)
    circles_img = cv2.erode(circles_img, np.ones((21, 21)))

    contours, _ = cv2.findContours(circles_img, 1, 2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        arclen = cv2.arcLength(cnt, True)
        if arclen == 0:
            continue
        circularity = (4 * np.pi * area) / (arclen * arclen)
        if (circularity < .75):
            cv2.drawContours(circles_img, [cnt], -1, 0, -1)

    # contour the image:
    components = {}
    contours, _ = cv2.findContours(img, 1, 2)
    for cnt in contours:
        validContour = False
        my_mid = None
        for mid in midpoints:
            if cv2.pointPolygonTest(cnt, mid, False) >= 0:
                validContour = True
                my_mid = mid
        if validContour:
            x, y, w, h = cv2.boundingRect(cnt)
            components[my_mid] = components.get(my_mid, []) + [(original_img[y:y+h, x:x+w],
                                                                circles_img[y:y+h, x:x+w],
                                                                post_sans_dilate_img[y:y+h, x:x+w])]

    to_ret = []
    for m in midpoints:
        if m not in components:
            to_ret.append('undefined')
        else:
            min_contour = min(
                components[m], key=lambda x: x[0].shape[0] * x[0].shape[1])
            to_ret.append(identify_component(*min_contour, orientations[m]))
            # print(to_ret[-1])
            # show_imgs(min_contour[0])

    return to_ret


def dist(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def show_imgs(*imgs, names=None):
    if __name__ != '__main__': return

    for e, img in enumerate(imgs):
        name = str(e) if names is None else names[e]
        cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process(img):
    img = resize_image(img)
    post_img, post_sans_dilate_img = clean_image(img)
    corners, line_segments = detect_graph_components(post_img)
    graph = build_graph(corners, line_segments)
    cedges, line_pairs = component_edges(
        graph, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    components = classify_components(post_img, cedges, graph, post_sans_dilate_img)
    circuit = build_circuit(graph, cedges, line_pairs, components)
    return circuit


if __name__ == "__main__":
    for i in range(1,10):
        img = cv2.imread("imgs/{}.JPG".format(i), 0)
        img = resize_image(img)
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        post_img, post_sans_dilate_img = clean_image(img)
        corners, line_segments = detect_graph_components(post_img)
        graph = build_graph(corners, line_segments)
        cedges, line_pairs = component_edges(
            graph, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        components = classify_components(post_img, cedges, graph, post_sans_dilate_img)
        circuit = build_circuit(graph, cedges, line_pairs, components)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                  (255, 255, 0), (0, 255, 255), (255, 0, 255)]

        raw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for corner in corners:
            cv2.circle(raw_img, corner, 6, colors[0], -1)
        for line in line_segments:
            cv2.line(raw_img, tuple(line[0]), tuple(line[1]), colors[1], 2)

        visualize = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for e, (line, lp) in enumerate(zip(cedges, line_pairs)):
            color = colors[e % len(colors)]
            cv2.line(visualize, tuple(graph[line[0]].loc), tuple(
                graph[line[1]].loc), color, 2)
            # for l in lp:
               # cv2.line(visualize, tuple(graph[l[0]].loc), tuple(
                   # graph[l[1]].loc), color, 2)

        print(components)
        # show_imgs(raw_img, names=[str(i)])
        show_imgs(raw_img, visualize, names=["raw", str(i)])
