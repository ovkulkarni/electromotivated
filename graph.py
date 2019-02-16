import cv2
import numpy as np
from collections import deque, namedtuple
from unionfind import UnionFind
from identify import identify_component

def dist(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


class Vertex:
    def __init__(self, index, loc, adjs=None):
        self.index = index
        self.loc = loc
        self.adjs = adjs if adjs is not None else set()

    def __repr__(self):
        return str(self.adjs)


def build_graph(raw_corners, segments):
    vs = []

    uf = UnionFind(list(range(len(raw_corners))))
    for a in range(len(raw_corners)):
        for b in range(len(raw_corners)):
            dis = dist(*raw_corners[a], *raw_corners[b])
            if dis < 10:
                uf.union(a, b)

    corners = []
    for conn in uf.components():
        cx = sum(raw_corners[i][0] for i in conn) // len(conn)
        cy = sum(raw_corners[i][1] for i in conn) // len(conn)
        corners.append((cx, cy))
    corners = raw_corners

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
            if min_corner_dist <= 75:
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
    for e, conn in enumerate(connecteds):
        for v in conn:
            color = colors[e % len(colors)]
            cv2.circle(img, graph[v].loc, 6, color, -1)
    # show_imgs(img)

    def edge_depth(edge):
        a, b = tuple(edge)
        return min(depth[a], depth[b])

    def edge_lset(edge):
        a, b = tuple(edge)
        return leaf_set[a] | leaf_set[b]

    all_cedges = []
    all_line_pairs = []
    for la in all_edges:
        for lb in all_edges:
            if edge_lset(la) & edge_lset(lb) or lb[0]*lb[1] >= la[0]*la[1]:
                continue

            # find min edge between these two lines
            min_edge, min_edge_dist = get_min_edge(graph, la, lb)

            # test for bad edges
            bad = False
            for pt in min_edge:
                if depth[pt] >= 2:
                    bad = True
                    break
                for a in graph[pt].adjs:
                    if a in (la + lb):
                        continue
                    if a in min_edge or colinear(graph, (pt, a), min_edge):
                        bad = True
                        break

            if min_edge_dist > 300 or bad:
                continue

            if colinear(graph, la, lb):
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


def slope_intercept(line, graph):
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


def get_min_edge(graph, la, lb):
    min_edge = None
    min_edge_dist = None
    for pta in la:
        for ptb in lb:
            dis = dist(*graph[pta].loc, *graph[ptb].loc)
            if min_edge_dist is None or dis < min_edge_dist:
                min_edge_dist = dis
                min_edge = (pta, ptb)

    return min_edge, min_edge_dist


def colinear(graph, la, lb):
    va, sa, ia = slope_intercept(la, graph)
    vb, sb, ib = slope_intercept(lb, graph)

    if va != vb:
        return False

    # find min edge between these two lines
    min_edge, min_edge_dist = get_min_edge(graph, la, lb)

    if va:
        dis = abs(graph[min_edge[0]].loc[0] - sb *
                  graph[min_edge[0]].loc[1] - ib)/(sb**2 + 1)**0.5
    else:
        dis = abs(graph[min_edge[0]].loc[1] - sb *
                  graph[min_edge[0]].loc[0] - ib) / (sb ** 2 + 1) ** 0.5

    return (min_edge_dist < 5 or dis/min_edge_dist < 0.5) and abs(sa - sb) ** 2 < 0.1


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

def show_imgs(*imgs, names=None):
    for e, img in enumerate(imgs):
        name = str(e) if names is None else names[e]
        cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()