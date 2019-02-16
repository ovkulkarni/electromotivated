import cv2
import numpy as np
from itertools import combinations


def resize_image(img):
    scale = np.sqrt(3e5 / (img.shape[0] * img.shape[1]))
    new_bounds = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    img = cv2.resize(img, new_bounds, interpolation=cv2.INTER_CUBIC)
    return img


def clean_image(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 25, 50)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((7, 7)))
    img = cv2.dilate(img, np.ones((5, 5)), iterations=2)
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

    # find corners
    corner_img = img*0
    corner_img[resps > threshold * resps.max()] = 255
    corner_contours, _ = cv2.findContours(corner_img, 1, 2)
    corners = []
    for cnt in corner_contours:
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01'] / M['m00'])
        corners.append((cx, cy))

    # find lines
    contours, _ = cv2.findContours(img, 1, 2)
    line_segments = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        p1 = box[0]
        p2 = min(box[1:], key=lambda x: np.linalg.norm(p1-x))

        p3, p4 = [p for p in box if not np.array_equal(
            p, p1) and not np.array_equal(p, p2)]

        line_segments.append((np.int0((p1 + p2) / 2), np.int0((p3 + p4) / 2)))

    return corners, line_segments


def dist(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


class Vertex:
    def __init__(self, index, loc, adjs=None):
        self.index = index
        self.loc = loc
        self.adjs = adjs if adjs is not None else set()

    def __repr__(self):
        return str(self.adjs)


def build_graph(corners, segments):
    vs = []
    for c in corners:
        v = Vertex(len(vs), c)
        vs.append(v)

    for seg in segments:
        seg_len = dist(*seg[0], *seg[1])
        seg_corners = []
        for pnt in seg:
            min_corner = None
            min_corner_dist = None
            for e, corner in enumerate(corners):
                dis = dist(*pnt, *corner)
                if not min_corner_dist or dis < min_corner_dist:
                    min_corner = e
                    min_corner_dist = dis
            if min_corner_dist <= 50:
                seg_corners.append(min_corner)
        if len(seg_corners) == 2:
            a, b = tuple(seg_corners)
            vs[a].adjs.add(b)
            vs[b].adjs.add(a)

    return vs


def broken_components(graph):
    vset = set(range(len(graph)))
    connecteds = []
    while vset:
        init = vset.pop()
        conn = set([init])
        q = [init]
        while q:
            v = graph[q.pop()]
            for a in v.adjs:
                if a in vset:
                    q.append(a)
                    conn.add(a)
                    vset.remove(a)
        if len(conn) >= 3:
            connecteds.append(conn)

    
    return connecteds


def show_imgs(*imgs):
    for e, img in enumerate(imgs):
        cv2.imshow(str(e), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    for i in range(1, 7):
        img = cv2.imread("imgs/{}.JPG".format(i), 0)
        img = resize_image(img)
        post_img = clean_image(img)
        corners, line_segments = detect_graph_components(post_img)
        graph = build_graph(corners, line_segments)
        components = broken_components(graph)
        #print(components)

        line_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                  (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        for e, c in enumerate(components):
            color = colors[e % len(colors)]
            for v in c:
                cv2.circle(line_img, graph[v].loc, 6, color, -1)

        show_imgs(img, line_img)
