import cv2
import numpy as np
from collections import deque, namedtuple
from unionfind import UnionFind
from identify import identify_component


def resize_image(img):
    scale = np.sqrt(4e5 / (img.shape[0] * img.shape[1]))
    new_bounds = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    img = cv2.resize(img, new_bounds, interpolation=cv2.INTER_CUBIC)
    return img


def clean_image(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 25, 30)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((7, 7)))
    img = cv2.dilate(img, np.ones((5, 5)), iterations=2)
    return img


def detect_graph_components(img):
    block_size = 6
    aperture = 15
    free_parameter = 0.04

    eroded_img = cv2.erode(img, np.ones((9, 9)), iterations=1)

    resps = cv2.cornerHarris(eroded_img, block_size, aperture, free_parameter)
    threshold = 0.1

    corner_img = img*0
    corner_img[resps > threshold * resps.max()] = 255

    resistor_img = img*0
    resistor_img[resps > threshold * resps.max()] = 255

    circles_img = np.copy(img)

    line_img = np.copy(eroded_img)
    resps = cv2.dilate(resps, np.ones((9, 9)), iterations=1)
    line_img[resps > threshold * resps.max()] = 0.

    # find corners
    corner_img = cv2.dilate(corner_img, np.ones((2, 2)), iterations=1)
    corner_img = cv2.morphologyEx(corner_img, cv2.MORPH_CLOSE, np.ones((5, 5)))

    corner_contours, _ = cv2.findContours(corner_img, 1, 2)
    corners = []
    for cnt in corner_contours:
        if cv2.contourArea(cnt) < 500:
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01'] / M['m00'])
            corners.append((cx, cy))
        else:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            for i, j in zip([0, 1, 2, 3], [1, 2, 3, 0]):
                corners.append(tuple(np.int0((box[i] + box[j]) / 2)))

    # find the resistors:
    resistor_img = cv2.morphologyEx(
        resistor_img, cv2.MORPH_CLOSE, np.ones((11, 11)), iterations=3)
    resistor_img = cv2.morphologyEx(
        resistor_img, cv2.MORPH_OPEN, np.ones((9, 9)), iterations=2)
    resistor_img = cv2.morphologyEx(
        resistor_img, cv2.MORPH_OPEN, np.ones((21, 21)), iterations=1)

    # find the circles:
    circles_img = cv2.morphologyEx(
        circles_img, cv2.MORPH_CLOSE, np.ones((5, 5)), iterations=5)
    circles_img = cv2.morphologyEx(
        circles_img, cv2.MORPH_OPEN, np.ones((20, 20)), iterations=1)
    contours, _ = cv2.findContours(circles_img, 1, 2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        arclen = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (arclen * arclen)
        if (circularity < .85):
            cv2.drawContours(circles_img, [cnt], -1, 0, -1)

    _, line_img = cv2.threshold(
        line_img-resistor_img-circles_img, 2, 255, cv2.THRESH_BINARY)

    # find lines
    contours, _ = cv2.findContours(line_img, 1, 2)
    line_segments = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
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


def classify_components(img, cedges, graph):
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

    for loc in locs:  # connect the contours completely
        cv2.line(img, loc[0], loc[1], 255, 10)
        cv2.circle(img, loc[0], 15, 0, -1)
        cv2.circle(img, loc[1], 15, 0, -1)

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
            components[my_mid] = identify_component(
                original_img[y:y+h, x:x+w], orientations[my_mid])
    return [components.get(m, 'undefined') for m in midpoints]


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


def component_edges(graph, img=None):
    # print(graph)
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
        if len(conn) >= 2:
            connecteds.append(conn)

    depth = {}
    parents = {}
    children = {}
    leaf_set = {}

    for conn in connecteds:
        leaves = set()
        for v in conn:
            if len(graph[v].adjs) == 1:
                leaves.add(v)
                depth[v] = 0
                leaf_set[v] = set([v])
                children[v] = set()
                parents[v] = set()

        q = deque()
        for leaf in leaves:
            q.append(leaf)
        while q:
            v = q.popleft()
            for a in graph[v].adjs:
                if a in depth:
                    if depth[a] > depth[v]:
                        leaf_set[a].update(leaf_set[v])
                        parents[a].add(v)
                else:
                    parents[a] = set([v])
                    leaf_set[a] = leaf_set[v] | {a}
                    depth[a] = depth[v] + 1
                    children[a] = set()
                    children[v].add(a)
                    q.append(a)

    all_edges = set()
    for v in set().union(*connecteds):
        for a in graph[v].adjs:
            all_edges.add(frozenset([v, a]))
    all_edges = {tuple(e) for e in all_edges}

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for e, line in enumerate(all_edges):
        color = colors[e % len(colors)]
        cv2.line(img, tuple(graph[line[0]].loc), tuple(
            graph[line[1]].loc), color, 2)
    #show_imgs(img)

    def edge_depth(edge):
        a, b = tuple(edge)
        return min(depth[a], depth[b])

    def edge_lset(edge):
        a, b = tuple(edge)
        return leaf_set[a] | leaf_set[b]

    all_cedges = []
    all_line_pairs = []
    for la in all_edges:
        va, sa, ia = slope_intercept(la)
        for lb in all_edges:
            if edge_lset(la) & edge_lset(lb) or lb[0]*lb[1] >= la[0]*la[1]:
                continue

            vb, sb, ib = slope_intercept(lb)
            if va != vb:
                continue

            # find min edge between these two lines
            min_edge = None
            min_edge_dist = None
            for pta in la:
                for ptb in lb:
                    dis = dist(*graph[pta].loc, *graph[ptb].loc)
                    if min_edge_dist is None or dis < min_edge_dist:
                        min_edge_dist = dis
                        min_edge = (pta, ptb)

            if min_edge_dist > 300:
                continue

            # determine if its a valid edge
            if va:
                dis = abs(graph[min_edge[0]].loc[0] - sb *
                          graph[min_edge[0]].loc[1] - ib)/(sb**2 + 1)**0.5
            else:
                dis = abs(graph[min_edge[0]].loc[1] - sb *
                          graph[min_edge[0]].loc[0] - ib)/(sb**2 + 1)**0.5
            if dis/min_edge_dist < 0.5 and abs(sa - sb) ** 2 < 0.1:
                all_cedges.append(tuple(min_edge))
                all_line_pairs.append((tuple(la), tuple(lb)))

    ordered_clp = []
    for c, lp in zip(all_cedges, all_line_pairs):
        dis = dist(*graph[c[0]].loc, *graph[c[1]].loc)
        ordered_clp.append((dis, c, lp))

    cedges = []
    line_pairs = []
    used_nodes = set()
    # print(sorted(ordered_clp))
    for _, c, lp in sorted(ordered_clp):
        if used_nodes & set(c):
            continue
        # print(used_nodes)
        # print(c)
        cedges.append(c)
        line_pairs.append(lp)
        used_nodes |= (edge_lset(c) - set(lp[0] + lp[1])) | set(c)

    # print()
    return cedges, line_pairs


def slope_intercept(line):
    line = tuple(line)
    x1, y1 = tuple(graph[line[0]].loc)
    x2, y2 = tuple(graph[line[1]].loc)

    vert = abs(y2 - y1) > abs(x2 - x1)
    if vert:
        slope = (x2 - x1) / (y2 - y1)
        intercept = x1 - slope*y1
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope*x1
    return vert, slope, intercept


class Node:
    def __init__(self, index, component, loc=None, adjs=None):
        self.index = index
        self.loc = loc
        self.component = component
        self.adjs = adjs if adjs is not None else []

    def __repr__(self):
        return "({} [{}]: {})".format(self.index, self.component, self.adjs)


def build_circuit(graph, cedges, line_pairs, components):
    ns = []
    index_map = {}
    reverse_index_map = {}

    def get_corner(o_ix):
        if o_ix not in index_map:
            corner_node = Node(len(ns), "corner", graph[o_ix].loc)
            index_map[o_ix] = len(ns)
            reverse_index_map[len(ns)] = o_ix
            ns.append(corner_node)
            return corner_node
        else:
            return ns[index_map[o_ix]]

    q = deque()
    for e, (cedge, comp_name) in enumerate(zip(cedges, components)):
        node = Node(len(ns), comp_name)
        ns.append(node)
        node.adjs = []
        for pt in cedge:
            corner_node = get_corner(pt)
            for line in line_pairs[e]:
                if pt in line:
                    next_pt = line[0] if line[0] != pt else line[1]
                    next_corner = get_corner(next_pt)
                    corner_node.adjs = [node.index, next_corner.index]
                    q.append(next_corner.index)
            node.adjs.append(index_map[pt])

    while q:
        node = ns[q.popleft()]
        o_ix = reverse_index_map[node.index]
        for a in graph[o_ix].adjs:
            add_to_q = (a not in index_map)
            adj_corner = get_corner(a)
            node.adjs.append(adj_corner.index)
            if add_to_q:
                q.append(adj_corner.index)

    return ns


def show_imgs(*imgs):
    for e, img in enumerate(imgs):
        cv2.imshow(str(e), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process(img):
    img = resize_image(img)
    post_img = clean_image(img)
    corners, line_segments = detect_graph_components(post_img)
    graph = build_graph(corners, line_segments)
    cedges, line_pairs = component_edges(graph)
    components = classify_components(post_img, cedges, graph)
    circuit = build_circuit(graph, cedges, line_pairs, components)
    return circuit


if __name__ == "__main__":
    for i in range(6, 18):
        img = cv2.imread("imgs/{}.JPG".format(i), 0)
        img = resize_image(img)
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        post_img = clean_image(img)
        corners, line_segments = detect_graph_components(post_img)
        graph = build_graph(corners, line_segments)
        cedges, line_pairs = component_edges(
            graph, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        components = classify_components(post_img, cedges, graph)
        circuit = build_circuit(graph, cedges, line_pairs, components)

        visualize = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                  (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        for e, (line, lp) in enumerate(zip(cedges, line_pairs)):
            color = colors[e % len(colors)]
            cv2.line(visualize, tuple(graph[line[0]].loc), tuple(
                graph[line[1]].loc), color, 2)
            #for l in lp:
            #    cv2.line(visualize, tuple(graph[l[0]].loc), tuple(
            #        graph[l[1]].loc), color, 2)

        show_imgs(visualize)
        # print(components)
