import SchemDraw as schem
import SchemDraw.elements as e

from cv2 import imread
from process_image import process
import math
import matplotlib
matplotlib.use('Agg')


def align_points(out, k):
    out.sort(key=lambda l: l.loc[k] if l.loc else 9999999)
    working = []
    for i in range(len(out)):
        if not out[i].loc:
            break
        working.append((i, out[i]))
        k_coords = [node.loc[k] for index, node in working if node.loc]
        if max(k_coords) - min(k_coords) <= 75:
            continue
        avg = sum(node.loc[k]
                  for index, node in working[:-1]) // len(working[:-1])
        for index, node in working[:-1]:
            out[index].loc[k] = avg
        working = [working[-1]]
    if working:
        avg = sum(node.loc[k] for index, node in working) // len(working)
        for index, node in working:
            out[index].loc[k] = avg


def render_image(fn, out_fn):
    img = imread(fn, 0)
    out = process(img)
    d = schem.Drawing(unit=1)
    max_y, min_y = max(x.loc[1] if x.loc else 0 for x in out), min(
        x.loc[1] if x.loc else float('inf') for x in out)
    max_x, min_x = max(x.loc[0] if x.loc else 0 for x in out), min(
        x.loc[0] if x.loc else float('inf') for x in out)

    y_range = max_y - min_y
    x_range = max_x - min_x

    WIDTH = 8
    HEIGHT = math.floor(WIDTH * (y_range / x_range))

    for node in out:
        node.loc = list(node.loc) if node.loc else None
        if node.component != "corner":
            connections = [out[x] for x in node.adjs]
            node.is_horizontal = abs(connections[1].loc[0] - connections[0].loc[0]) > abs(
                connections[1].loc[1] - connections[0].loc[1])

    save = {i: out[i] for i in range(len(out))}
    align_points(out, 0)
    align_points(out, 1)
    out = [save[i] for i in save]
    for node in out:
        conns = [out[x] for x in node.adjs if out[x].component != "corner"]
        for component in conns:
            other = out[list(
                filter(lambda n: out[n] != node, component.adjs))[0]]
            if other.loc == node.loc:
                if component.is_horizontal:
                    if other.loc[0] > node.loc[0]:
                        node.loc[0] -= 40
                    else:
                        node.loc[0] += 40
                else:
                    if other.loc[1] > node.loc[1]:
                        node.loc[1] += 40
                    else:
                        node.loc[1] -= 40
    for node in out:
        if node.loc:
            node.loc[0] = 0.5 + (node.loc[0] - min_x) * (WIDTH / x_range)
            node.loc[1] = HEIGHT - \
                (0.5 + (node.loc[1] - min_y) * (HEIGHT / y_range))
    counts = {
        'resistor': 1,
        'capacitor': 1
    }
    for node in out:
        connections = [out[x] for x in node.adjs]
        if not node.loc:
            if node.is_horizontal:
                start = min(connections, key=lambda l: l.loc[0])
                end = max(connections, key=lambda l: l.loc[0])
            else:
                start = min(connections, key=lambda l: l.loc[1])
                end = max(connections, key=lambda l: l.loc[1])
            if node.component == "resistor/inductor":
                d.add(e.RES, xy=start.loc, to=end.loc,
                      label='R{}'.format(counts['resistor']))
                counts['resistor'] += 1
            if node.component == "rightbattery" or node.component == "bottombattery":
                d.add(e.BATTERY, xy=end.loc, to=start.loc)
            if node.component == "leftbattery" or node.component == "topbattery":
                d.add(e.BATTERY, xy=start.loc, to=end.loc)
            if node.component == "capacitor":
                d.add(e.CAP, xy=start.loc, to=end.loc,
                      label='R{}'.format(counts['capacitor']))
                counts['capacitor'] += 1
        else:
            for conn in connections:
                if conn.loc:
                    d.add(e.LINE, endpts=[node.loc, conn.loc])
    d.draw()
    d.save(out_fn)
