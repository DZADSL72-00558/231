"""
Hybrid A* with trailer
@author: Huiming Zhou
"""

import os
import sys
import math
import heapq
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.kdtree as kd

sys.path.append(os.path.dirname(os.path.abspath('__file__')) +
                "/../../MotionPlanning/")

import HybridAstarPlanner.astar as astar
import HybridAstarPlanner.draw as draw
import CurvesGenerator.reeds_shepp as rs


class C:  # Parameter config
    PI = np.pi

    XY_RESO = 2.0  # [m]
    YAW_RESO = np.deg2rad(15.0)  # [rad]
    GOAL_YAW_ERROR = np.deg2rad(3.0)  # [rad]
    MOVE_STEP = 0.2  # [m] path interporate resolution
    N_STEER = 20.0  # number of steer command
    COLLISION_CHECK_STEP = 10  # skip number for collision check
    EXTEND_AREA = 5.0  # [m] map extend length

    GEAR_COST = 100.0  # switch back penalty cost
    BACKWARD_COST = 5.0  # backward penalty cost
    STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
    STEER_ANGLE_COST = 1.0  # steer angle penalty cost
    SCISSORS_COST = 200.0  # scissors cost
    H_COST = 10.0  # Heuristic cost

    W = 3.0  # [m] width of vehicle
    WB = 3.5  # [m] wheel base: rear to front steer
    WD = 0.7 * W  # [m] distance between left-right wheels
    RF = 4.5  # [m] distance from rear to vehicle front end of vehicle
    RB = 1.0  # [m] distance from rear to vehicle back end of vehicle

    RTR = 8.0  # [m] rear to trailer wheel
    RTF = 1.0  # [m] distance from rear to vehicle front end of trailer
    RTB = 9.0  # [m] distance from rear to vehicle back end of trailer
    TR = 0.5  # [m] tyre radius
    TW = 1.0  # [m] tyre width

    
    MAX_STEER = np.deg2rad(45.0)  # max steering angle [rad]
    steer_change_max = np.deg2rad(30.0)  # maximum steering speed [rad/s]
    speed_max = 55.0 / 3.6  # maximum speed [m/s]
    speed_min = -20.0 / 3.6  # minimum speed [m/s]
    acceleration_max = 1.0  # maximum acceleration [m/s2]

class Node:
    def __init__(self, xind, yind, yawind, direction, x, y,
                 yaw, yawt, directions, steer, cost, pind):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yawt = yawt
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind


class Para:
    def __init__(self, minx, miny, minyaw, minyawt, maxx, maxy, maxyaw, maxyawt,
                 xw, yw, yaww, yawtw, xyreso, yawreso, ox, oy, kdtree):
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.minyawt = minyawt
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.maxyawt = maxyawt
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.yawtw = yawtw
        self.xyreso = xyreso
        self.yawreso = yawreso
        self.ox = ox
        self.oy = oy
        self.kdtree = kdtree


class Path:
    def __init__(self, x, y, yaw, yawt, direction, cost):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yawt = yawt
        self.direction = direction
        self.cost = cost


class QueuePrior:
    def __init__(self):
        self.queue = []

    def empty(self):
        return len(self.queue) == 0  # if Q is empty

    def put(self, item, priority):
        heapq.heappush(self.queue, (priority, item))  # reorder x using priority

    def get(self):
        return heapq.heappop(self.queue)[1]  # pop out element with smallest priority


def hybrid_astar_planning(sx, sy, syaw, syawt, gx, gy,
                          gyaw, gyawt, ox, oy, xyreso, yawreso):
    """
    planning hybrid A* path.
    :param sx: starting node x position [m]
    :param sy: starting node y position [m]
    :param syaw: starting node yaw angle [rad]
    :param syawt: starting node trailer yaw angle [rad]
    :param gx: goal node x position [m]
    :param gy: goal node y position [m]
    :param gyaw: goal node yaw angle [rad]
    :param gyawt: goal node trailer yaw angle [rad]
    :param ox: obstacle x positions [m]
    :param oy: obstacle y positions [m]
    :param xyreso: grid resolution [m]
    :param yawreso: yaw resolution [m]
    :return: hybrid A* path
    """

    sxr, syr = round(sx / xyreso), round(sy / xyreso)
    gxr, gyr = round(gx / xyreso), round(gy / xyreso)
    syawr = round(rs.pi_2_pi(syaw) / yawreso)
    gyawr = round(rs.pi_2_pi(gyaw) / yawreso)

    nstart = Node(sxr, syr, syawr, 1, [sx], [sy], [syaw], [syawt], [1], 0.0, 0.0, -1)
    ngoal = Node(gxr, gyr, gyawr, 1, [gx], [gy], [gyaw], [gyawt], [1], 0.0, 0.0, -1)

    kdtree = kd.KDTree([[x, y] for x, y in zip(ox, oy)])
    P = calc_parameters(ox, oy, xyreso, yawreso, kdtree)

    hmap = astar.calc_holonomic_heuristic_with_obstacle(ngoal, P.ox, P.oy, P.xyreso, 1.0)
    steer_set, direc_set = calc_motion_set()
    open_set, closed_set = {calc_index(nstart, P): nstart}, {}

    qp = QueuePrior()
    qp.put(calc_index(nstart, P), calc_hybrid_cost(nstart, hmap, P))

    while True:
        if not open_set:
            return None

        ind = qp.get()
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        update, fpath = update_node_with_analystic_expantion(n_curr, ngoal, gyawt, P)

        if update:
            fnode = fpath
            break

        yawt0 = n_curr.yawt[0]

        for i in range(len(steer_set)):
            node = calc_next_node(n_curr, ind, steer_set[i], direc_set[i], P)

            if not is_index_ok(node, yawt0, P):
                continue

            node_ind = calc_index(node, P)

            if node_ind in closed_set:
                continue

            if node_ind not in open_set:
                open_set[node_ind] = node
                qp.put(node_ind, calc_hybrid_cost(node, hmap, P))
            else:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind] = node

    print("final expand node: ", len(open_set) + len(closed_set))

    return extract_path(closed_set, fnode, nstart)


def extract_path(closed, ngoal, nstart):
    rx, ry, ryaw, ryawt, direc = [], [], [], [], []
    cost = 0.0
    node = ngoal

    while True:
        rx += node.x[::-1]
        ry += node.y[::-1]
        ryaw += node.yaw[::-1]
        ryawt += node.yawt[::-1]
        direc += node.directions[::-1]
        cost += node.cost

        if is_same_grid(node, nstart):
            break

        node = closed[node.pind]

    rx = rx[::-1]
    ry = ry[::-1]
    ryaw = ryaw[::-1]
    ryawt = ryawt[::-1]
    direc = direc[::-1]

    direc[0] = direc[1]
    path = Path(rx, ry, ryaw, ryawt, direc, cost)

    return path


def update_node_with_analystic_expantion(n_curr, ngoal, gyawt, P):
    path = analystic_expantion(n_curr, ngoal, P)  # rs path: n -> ngoal

    if not path:
        return False, None

    steps = [C.MOVE_STEP * d for d in path.directions]
    yawt = calc_trailer_yaw(path.yaw, n_curr.yawt[-1], steps)

    if abs(rs.pi_2_pi(yawt[-1] - gyawt)) >= C.GOAL_YAW_ERROR:
        return False, None

    fx = path.x[1:-1]
    fy = path.y[1:-1]
    fyaw = path.yaw[1:-1]

    fd = []
    for d in path.directions[1:-1]:
        if d >= 0:
            fd.append(1.0)
        else:
            fd.append(-1.0)
    # fd = path.directions[1:-1]

    fcost = n_curr.cost + calc_rs_path_cost(path, yawt)
    fpind = calc_index(n_curr, P)
    fyawt = yawt[1:-1]
    fsteer = 0.0

    fpath = Node(n_curr.xind, n_curr.yind, n_curr.yawind, n_curr.direction,
                 fx, fy, fyaw, fyawt, fd, fsteer, fcost, fpind)

    return True, fpath


def analystic_expantion(node, ngoal, P):
    sx, sy, syaw = node.x[-1], node.y[-1], node.yaw[-1]
    gx, gy, gyaw = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1]

    maxc = math.tan(C.MAX_STEER) / C.WB
    paths = rs.calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=C.MOVE_STEP)

    if not paths:
        return None

    pq = QueuePrior()
    for path in paths:
        steps = [C.MOVE_STEP * d for d in path.directions]
        yawt = calc_trailer_yaw(path.yaw, node.yawt[-1], steps)
        pq.put(path, calc_rs_path_cost(path, yawt))

    # while not pq.empty():
    path = pq.get()
    steps = [C.MOVE_STEP * d for d in path.directions]
    yawt = calc_trailer_yaw(path.yaw, node.yawt[-1], steps)
    ind = range(0, len(path.x), C.COLLISION_CHECK_STEP)

    pathx = [path.x[k] for k in ind]
    pathy = [path.y[k] for k in ind]
    pathyaw = [path.yaw[k] for k in ind]
    pathyawt = [yawt[k] for k in ind]

    if not is_collision(pathx, pathy, pathyaw, pathyawt, P):
        return path

    return None


def calc_next_node(n, ind, u, d, P):
    step = C.XY_RESO * 2.0

    nlist = math.ceil(step / C.MOVE_STEP)
    xlist = [n.x[-1] + d * C.MOVE_STEP * math.cos(n.yaw[-1])]
    ylist = [n.y[-1] + d * C.MOVE_STEP * math.sin(n.yaw[-1])]
    yawlist = [rs.pi_2_pi(n.yaw[-1] + d * C.MOVE_STEP / C.WB * math.tan(u))]
    yawtlist = [rs.pi_2_pi(n.yawt[-1] +
                           d * C.MOVE_STEP / C.RTR * math.sin(n.yaw[-1] - n.yawt[-1]))]

    for i in range(nlist - 1):
        xlist.append(xlist[i] + d * C.MOVE_STEP * math.cos(yawlist[i]))
        ylist.append(ylist[i] + d * C.MOVE_STEP * math.sin(yawlist[i]))
        yawlist.append(rs.pi_2_pi(yawlist[i] + d * C.MOVE_STEP / C.WB * math.tan(u)))
        yawtlist.append(rs.pi_2_pi(yawtlist[i] +
                                   d * C.MOVE_STEP / C.RTR * math.sin(yawlist[i] - yawtlist[i])))

    xind = round(xlist[-1] / P.xyreso)
    yind = round(ylist[-1] / P.xyreso)
    yawind = round(yawlist[-1] / P.yawreso)

    cost = 0.0

    if d > 0:
        direction = 1.0
        cost += abs(step)
    else:
        direction = -1.0
        cost += abs(step) * C.BACKWARD_COST

    if direction != n.direction:  # switch back penalty
        cost += C.GEAR_COST

    cost += C.STEER_ANGLE_COST * abs(u)  # steer penalyty
    cost += C.STEER_CHANGE_COST * abs(n.steer - u)  # steer change penalty
    cost += C.SCISSORS_COST * sum([abs(rs.pi_2_pi(x - y))
                                   for x, y in zip(yawlist, yawtlist)])  # jacknif cost
    cost = n.cost + cost

    directions = [direction for _ in range(len(xlist))]

    node = Node(xind, yind, yawind, direction, xlist, ylist,
                yawlist, yawtlist, directions, u, cost, ind)

    return node


def is_collision(x, y, yaw, yawt, P):
    for ix, iy, iyaw, iyawt in zip(x, y, yaw, yawt):
        d = 0.5
        deltal = (C.RTF - C.RTB) / 2.0
        rt = (C.RTF + C.RTB) / 2.0 + d

        ctx = ix + deltal * math.cos(iyawt)
        cty = iy + deltal * math.sin(iyawt)

        idst = P.kdtree.query_ball_point([ctx, cty], rt)

        if idst:
            for i in idst:
                xot = P.ox[i] - ctx
                yot = P.oy[i] - cty

                dx_trail = xot * math.cos(iyawt) + yot * math.sin(iyawt)
                dy_trail = -xot * math.sin(iyawt) + yot * math.cos(iyawt)

                if abs(dx_trail) <= rt and \
                        abs(dy_trail) <= C.W / 2.0 + d:
                    return True

        deltal = (C.RF - C.RB) / 2.0
        rc = (C.RF + C.RB) / 2.0 + d

        cx = ix + deltal * math.cos(iyaw)
        cy = iy + deltal * math.sin(iyaw)

        ids = P.kdtree.query_ball_point([cx, cy], rc)

        if ids:
            for i in ids:
                xo = P.ox[i] - cx
                yo = P.oy[i] - cy

                dx_car = xo * math.cos(iyaw) + yo * math.sin(iyaw)
                dy_car = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

                if abs(dx_car) <= rc and \
                        abs(dy_car) <= C.W / 2.0 + d:
                    return True
        #
        # theta = np.linspace(0, 2 * np.pi, 200)
        # x1 = ctx + np.cos(theta) * rt
        # y1 = cty + np.sin(theta) * rt
        # x2 = cx + np.cos(theta) * rc
        # y2 = cy + np.sin(theta) * rc
        #
        # plt.plot(x1, y1, 'b')
        # plt.plot(x2, y2, 'g')

    return False


def calc_trailer_yaw(yaw, yawt0, steps):
    yawt = [0.0 for _ in range(len(yaw))]
    yawt[0] = yawt0

    for i in range(1, len(yaw)):
        yawt[i] += yawt[i - 1] + steps[i - 1] / C.RTR * math.sin(yaw[i - 1] - yawt[i - 1])

    return yawt


def trailer_motion_model(x, y, yaw, yawt, D, d, L, delta):
    x += D * math.cos(yaw)
    y += D * math.sin(yaw)
    yaw += D / L * math.tan(delta)
    yawt += D / d * math.sin(yaw - yawt)

    return x, y, yaw, yawt


def calc_rs_path_cost(rspath, yawt):
    cost = 0.0

    for lr in rspath.lengths:
        if lr >= 0:
            cost += 1
        else:
            cost += abs(lr) * C.BACKWARD_COST

    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
            cost += C.GEAR_COST

    for ctype in rspath.ctypes:
        if ctype != "S":
            cost += C.STEER_ANGLE_COST * abs(C.MAX_STEER)

    nctypes = len(rspath.ctypes)
    ulist = [0.0 for _ in range(nctypes)]

    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = -C.MAX_STEER
        elif rspath.ctypes[i] == "WB":
            ulist[i] = C.MAX_STEER

    for i in range(nctypes - 1):
        cost += C.STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    cost += C.SCISSORS_COST * sum([abs(rs.pi_2_pi(x - y))
                                   for x, y in zip(rspath.yaw, yawt)])

    return cost


def calc_motion_set():
    s = [i for i in np.arange(C.MAX_STEER / C.N_STEER,
                              C.MAX_STEER, C.MAX_STEER / C.N_STEER)]

    steer = [0.0] + s + [-i for i in s]
    direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
    steer = steer + steer

    return steer, direc


def calc_hybrid_cost(node, hmap, P):
    cost = node.cost + \
           C.H_COST * hmap[node.xind - P.minx][node.yind - P.miny]

    return cost


def calc_index(node, P):
    ind = (node.yawind - P.minyaw) * P.xw * P.yw + \
          (node.yind - P.miny) * P.xw + \
          (node.xind - P.minx)

    yawt_ind = round(node.yawt[-1] / P.yawreso)
    ind += (yawt_ind - P.minyawt) * P.xw * P.yw * P.yaww

    return ind


def is_index_ok(node, yawt0, P):
    if node.xind <= P.minx or \
            node.xind >= P.maxx or \
            node.yind <= P.miny or \
            node.yind >= P.maxy:
        return False

    steps = [C.MOVE_STEP * d for d in node.directions]
    yawt = calc_trailer_yaw(node.yaw, yawt0, steps)

    ind = range(0, len(node.x), C.COLLISION_CHECK_STEP)

    x = [node.x[k] for k in ind]
    y = [node.y[k] for k in ind]
    yaw = [node.yaw[k] for k in ind]
    yawt = [yawt[k] for k in ind]

    if is_collision(x, y, yaw, yawt, P):
        return False

    return True


def calc_parameters(ox, oy, xyreso, yawreso, kdtree):
    minxm = min(ox) - C.EXTEND_AREA
    minym = min(oy) - C.EXTEND_AREA
    maxxm = max(ox) + C.EXTEND_AREA
    maxym = max(oy) + C.EXTEND_AREA

    ox.append(minxm)
    oy.append(minym)
    ox.append(maxxm)
    oy.append(maxym)

    minx = round(minxm / xyreso)
    miny = round(minym / xyreso)
    maxx = round(maxxm / xyreso)
    maxy = round(maxym / xyreso)

    xw, yw = maxx - minx, maxy - miny

    minyaw = round(-C.PI / yawreso) - 1
    maxyaw = round(C.PI / yawreso)
    yaww = maxyaw - minyaw

    minyawt, maxyawt, yawtw = minyaw, maxyaw, yaww

    P = Para(minx, miny, minyaw, minyawt, maxx, maxy, maxyaw,
             maxyawt, xw, yw, yaww, yawtw, xyreso, yawreso, ox, oy, kdtree)

    return P


def is_same_grid(node1, node2):
    if node1.xind != node2.xind or \
            node1.yind != node2.yind or \
            node1.yawind != node2.yawind:
        return False

    return True


def draw_model(x, y, yaw, yawt, steer, color='black'):
    car = np.array([[-C.RB, -C.RB, C.RF, C.RF, -C.RB],
                    [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])

    trail = np.array([[-C.RTB, -C.RTB, C.RTF, C.RTF, -C.RTB],
                      [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])

    wheel = np.array([[-C.TR, -C.TR, C.TR, C.TR, -C.TR],
                      [C.TW / 4, -C.TW / 4, -C.TW / 4, C.TW / 4, C.TW / 4]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()
    rltWheel = wheel.copy()
    rrtWheel = wheel.copy()

    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])

    Rot2 = np.array([[math.cos(steer), -math.sin(steer)],
                     [math.sin(steer), math.cos(steer)]])

    Rot3 = np.array([[math.cos(yawt), -math.sin(yawt)],
                     [math.sin(yawt), math.cos(yawt)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[C.WB], [-C.WD / 2]])
    flWheel += np.array([[C.WB], [C.WD / 2]])
    rrWheel[1, :] -= C.WD / 2
    rlWheel[1, :] += C.WD / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    rltWheel += np.array([[-C.RTR], [C.WD / 2]])
    rrtWheel += np.array([[-C.RTR], [-C.WD / 2]])

    rltWheel = np.dot(Rot3, rltWheel)
    rrtWheel = np.dot(Rot3, rrtWheel)
    trail = np.dot(Rot3, trail)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    rrtWheel += np.array([[x], [y]])
    rltWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])
    trail += np.array([[x], [y]])

    plt.plot(car[0, :], car[1, :], color)
    plt.plot(trail[0, :], trail[1, :], color)
    plt.plot(frWheel[0, :], frWheel[1, :], color)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color)
    plt.plot(flWheel[0, :], flWheel[1, :], color)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color)
    plt.plot(rrtWheel[0, :], rrtWheel[1, :], color)
    plt.plot(rltWheel[0, :], rltWheel[1, :], color)
    draw.Arrow(x, y, yaw, C.WB * 0.8, color)

fator=5
totalwidth=20*fator
totalhight=9*fator
linewidth=4*fator
def design_obstacles():
    ox, oy = [], []

    for i in range(0, totalwidth+1):
        ox.append(i)
        oy.append(totalhight)

    for i in range(0, totalwidth+1):
        ox.append(i)
        oy.append(0)
    for i in range(0, totalhight+1):
        ox.append(0)
        oy.append(i)
    for i in range(2*linewidth,totalwidth+1):
        ox.append(i)
        oy.append(linewidth)
    for i in range(2*linewidth,totalwidth+1):
        ox.append(i)
        oy.append(totalhight-linewidth)
    for i in range(linewidth,totalhight-linewidth+1):
        ox.append(linewidth*2)
        oy.append(i)
    return ox, oy
#%%
'''
def test(x, y, yaw, yawt, ox, oy):
    d = 0.5
    deltal = (C.RTF - C.RTB) / 2.0
    rt = (C.RTF + C.RTB) / 2.0 + d

    ctx = x + deltal * math.cos(yawt)
    cty = y + deltal * math.sin(yawt)

    deltal = (C.RF - C.RB) / 2.0
    rc = (C.RF + C.RB) / 2.0 + d

    xot = ox - ctx
    yot = oy - cty

    dx_trail = xot * math.cos(yawt) + yot * math.sin(yawt)
    dy_trail = -xot * math.sin(yawt) + yot * math.cos(yawt)

    if abs(dx_trail) <= rt - d and \
            abs(dy_trail) <= C.W / 2.0:
        print("test1: Collision")
    else:
        print("test1: No collision")

    # test 2

    cx = x + deltal * math.cos(yaw)
    cy = y + deltal * math.sin(yaw)

    xo = ox - cx
    yo = oy - cy

    dx_car = xo * math.cos(yaw) + yo * math.sin(yaw)
    dy_car = -xo * math.sin(yaw) + yo * math.cos(yaw)

    if abs(dx_car) <= rc - d and \
            abs(dy_car) <= C.W / 2.0:
        print("test2: Collision")
    else:
        print("test2: No collision")

    theta = np.linspace(0, 2 * np.pi, 200)
    x1 = ctx + np.cos(theta) * rt
    y1 = cty + np.sin(theta) * rt
    x2 = cx + np.cos(theta) * rc
    y2 = cy + np.sin(theta) * rc

    plt.plot(x1, y1, 'b')
    plt.plot(x2, y2, 'g')
    plt.plot(ox, oy, 'sr')

    plt.plot([-rc, -rc, rc, rc, -rc],
             [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2])
    plt.plot([-rt, -rt, rt, rt, -rt],
             [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2])
    plt.plot(dx_car, dy_car, 'sr')
    plt.plot(dx_trail, dy_trail, 'sg')

'''
print("start!")

sx, sy = linewidth*2+C.RF, totalhight-linewidth/2  # [m]
syaw0 = np.deg2rad(180.0)
syawt = np.deg2rad(180.0)

gx, gy = linewidth*2+C.RF, linewidth/2  # [m]
gyaw0 = np.deg2rad(0.0)
gyawt = np.deg2rad(0.0)

ox, oy = design_obstacles()
plt.plot(ox, oy, 'sk')
draw_model(sx, sy, syaw0, syawt, 0.0)
draw_model(gx, gy, gyaw0, gyawt, 0.0)
plt.show()
#%%
# test(sx, sy, syaw0, syawt, 3.5, 32)
# plt.axis("equal")

oox, ooy = ox[:], oy[:]

t0 = time.time()
path = hybrid_astar_planning(sx, sy, syaw0, syawt, gx, gy, gyaw0, gyawt,
                             oox, ooy, C.XY_RESO, C.YAW_RESO)
#%%
#MPC
"""
Linear MPC controller (X-Y frame)
author: huiming zhou
"""

import os
import sys
import math
import cvxpy
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import Control.draw as draw
import CurvesGenerator.reeds_shepp as rs
import CurvesGenerator.cubic_spline as cs


class P:
    # System config
    NX = 4  # state vector: z = [x, y, v, phi]
    NU = 2  # input vector: u = [acceleration, steer]
    T = 6  # finite time horizon length

    # MPC config
    Q = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for states
    Qf = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for end state
    R = np.diag([0.01, 0.1])  # penalty for inputs
    Rd = np.diag([0.01, 0.1])  # penalty for change of inputs

    dist_stop = 1.5  # stop permitted when dist to goal < dist_stop
    speed_stop = 0.5 / 3.6  # stop permitted when speed < speed_stop
    time_max = 500.0  # max simulation time
    iter_max = 5  # max iteration
    target_speed = 10.0 / 3.6  # target speed
    N_IND = 10  # search index number
    dt = 0.2  # time step
    d_dist = 1.0  # dist step
    du_res = 0.1  # threshold for stopping iteration

    # vehicle config
    RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    W = 2.4  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 2.5  # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7  # [m] Tyre width

    steer_max = np.deg2rad(45.0)  # max steering angle [rad]
    steer_change_max = np.deg2rad(30.0)  # maximum steering speed [rad/s]
    speed_max = 55.0 / 3.6  # maximum speed [m/s]
    speed_min = -20.0 / 3.6  # minimum speed [m/s]
    acceleration_max = 1.0  # maximum acceleration [m/s2]


class Node:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, direct=1.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.direct = direct

    def update(self, a, delta, direct):
        delta = self.limit_input_delta(delta)
        self.x += self.v * math.cos(self.yaw) * P.dt
        self.y += self.v * math.sin(self.yaw) * P.dt
        self.yaw += self.v / P.WB * math.tan(delta) * P.dt
        self.direct = direct
        self.v += self.direct * a * P.dt
        self.v = self.limit_speed(self.v)

    @staticmethod
    def limit_input_delta(delta):
        if delta >= P.steer_max:
            return P.steer_max

        if delta <= -P.steer_max:
            return -P.steer_max

        return delta

    @staticmethod
    def limit_speed(v):
        if v >= P.speed_max:
            return P.speed_max

        if v <= P.speed_min:
            return P.speed_min

        return v


class PATH:
    def __init__(self, cx, cy, cyaw):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.length = len(cx)
        self.ind_old = 0

    def nearest_index(self, node):
        """
        calc index of the nearest node in N steps
        :param node: current information
        :return: nearest index, lateral distance to ref point
        """

        dx = [node.x - x for x in self.cx[self.ind_old: (self.ind_old + P.N_IND)]]
        dy = [node.y - y for y in self.cy[self.ind_old: (self.ind_old + P.N_IND)]]
        dist = np.hypot(dx, dy)

        ind_in_N = int(np.argmin(dist))
        ind = self.ind_old + ind_in_N
        self.ind_old = ind

        rear_axle_vec_rot_90 = np.array([[math.cos(node.yaw + math.pi / 2.0)],
                                         [math.sin(node.yaw + math.pi / 2.0)]])

        vec_target_2_rear = np.array([[dx[ind_in_N]],
                                      [dy[ind_in_N]]])

        er = np.dot(vec_target_2_rear.T, rear_axle_vec_rot_90)
        er = er[0][0]

        return ind, er


def calc_ref_trajectory_in_T_step(node, ref_path, sp):
    """
    calc referent trajectory in T steps: [x, y, v, yaw]
    using the current velocity, calc the T points along the reference path
    :param node: current information
    :param ref_path: reference path: [x, y, yaw]
    :param sp: speed profile (designed speed strategy)
    :return: reference trajectory
    """

    z_ref = np.zeros((P.NX, P.T + 1))
    length = ref_path.length

    ind, _ = ref_path.nearest_index(node)

    z_ref[0, 0] = ref_path.cx[ind]
    z_ref[1, 0] = ref_path.cy[ind]
    z_ref[2, 0] = sp[ind]
    z_ref[3, 0] = ref_path.cyaw[ind]

    dist_move = 0.0

    for i in range(1, P.T + 1):
        dist_move += abs(node.v) * P.dt
        ind_move = int(round(dist_move / P.d_dist))
        index = min(ind + ind_move, length - 1)

        z_ref[0, i] = ref_path.cx[index]
        z_ref[1, i] = ref_path.cy[index]
        z_ref[2, i] = sp[index]
        z_ref[3, i] = ref_path.cyaw[index]

    return z_ref, ind


def linear_mpc_control(z_ref, z0, a_old, delta_old):
    """
    linear mpc controller
    :param z_ref: reference trajectory in T steps
    :param z0: initial state vector
    :param a_old: acceleration of T steps of last time
    :param delta_old: delta of T steps of last time
    :return: acceleration and delta strategy based on current information
    """

    if a_old is None or delta_old is None:
        a_old = [0.0] * P.T
        delta_old = [0.0] * P.T

    x, y, yaw, v = None, None, None, None

    for k in range(P.iter_max):
        z_bar = predict_states_in_T_step(z0, a_old, delta_old, z_ref)
        a_rec, delta_rec = a_old[:], delta_old[:]
        a_old, delta_old, x, y, yaw, v = solve_linear_mpc(z_ref, z_bar, z0, delta_old)

        du_a_max = max([abs(ia - iao) for ia, iao in zip(a_old, a_rec)])
        du_d_max = max([abs(ide - ido) for ide, ido in zip(delta_old, delta_rec)])

        if max(du_a_max, du_d_max) < P.du_res:
            break

    return a_old, delta_old, x, y, yaw, v


def predict_states_in_T_step(z0, a, delta, z_ref):
    """
    given the current state, using the acceleration and delta strategy of last time,
    predict the states of vehicle in T steps.
    :param z0: initial state
    :param a: acceleration strategy of last time
    :param delta: delta strategy of last time
    :param z_ref: reference trajectory
    :return: predict states in T steps (z_bar, used for calc linear motion model)
    """

    z_bar = z_ref * 0.0

    for i in range(P.NX):
        z_bar[i, 0] = z0[i]

    node = Node(x=z0[0], y=z0[1], v=z0[2], yaw=z0[3])

    for ai, di, i in zip(a, delta, range(1, P.T + 1)):
        node.update(ai, di, 1.0)
        z_bar[0, i] = node.x
        z_bar[1, i] = node.y
        z_bar[2, i] = node.v
        z_bar[3, i] = node.yaw

    return z_bar


def calc_linear_discrete_model(v, phi, delta):
    """
    calc linear and discrete time dynamic model.
    :param v: speed: v_bar
    :param phi: angle of vehicle: phi_bar
    :param delta: steering angle: delta_bar
    :return: A, B, C
    """

    A = np.array([[1.0, 0.0, P.dt * math.cos(phi), - P.dt * v * math.sin(phi)],
                  [0.0, 1.0, P.dt * math.sin(phi), P.dt * v * math.cos(phi)],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, P.dt * math.tan(delta) / P.WB, 1.0]])

    B = np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [P.dt, 0.0],
                  [0.0, P.dt * v / (P.WB * math.cos(delta) ** 2)]])

    C = np.array([P.dt * v * math.sin(phi) * phi,
                  -P.dt * v * math.cos(phi) * phi,
                  0.0,
                  -P.dt * v * delta / (P.WB * math.cos(delta) ** 2)])

    return A, B, C


def solve_linear_mpc(z_ref, z_bar, z0, d_bar):
    """
    solve the quadratic optimization problem using cvxpy, solver: OSQP
    :param z_ref: reference trajectory (desired trajectory: [x, y, v, yaw])
    :param z_bar: predicted states in T steps
    :param z0: initial state
    :param d_bar: delta_bar
    :return: optimal acceleration and steering strategy
    """

    z = cvxpy.Variable((P.NX, P.T + 1))
    u = cvxpy.Variable((P.NU, P.T))

    cost = 0.0
    constrains = []

    for t in range(P.T):
        cost += cvxpy.quad_form(u[:, t], P.R)
        cost += cvxpy.quad_form(z_ref[:, t] - z[:, t], P.Q)

        A, B, C = calc_linear_discrete_model(z_bar[2, t], z_bar[3, t], d_bar[t])

        constrains += [z[:, t + 1] == A @ z[:, t] + B @ u[:, t] + C]

        if t < P.T - 1:
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], P.Rd)
            constrains += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= P.steer_change_max * P.dt]

    cost += cvxpy.quad_form(z_ref[:, P.T] - z[:, P.T], P.Qf)

    constrains += [z[:, 0] == z0]
    constrains += [z[2, :] <= P.speed_max]
    constrains += [z[2, :] >= P.speed_min]
    constrains += [cvxpy.abs(u[0, :]) <= P.acceleration_max]
    constrains += [cvxpy.abs(u[1, :]) <= P.steer_max]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constrains)
    prob.solve(solver=cvxpy.OSQP)

    a, delta, x, y, yaw, v = None, None, None, None, None, None

    if prob.status == cvxpy.OPTIMAL or \
            prob.status == cvxpy.OPTIMAL_INACCURATE:
        x = z.value[0, :]
        y = z.value[1, :]
        v = z.value[2, :]
        yaw = z.value[3, :]
        a = u.value[0, :]
        delta = u.value[1, :]
    else:
        print("Cannot solve linear mpc!")

    return a, delta, x, y, yaw, v


def calc_speed_profile(cx, cy, cyaw, target_speed):
    """
    design appropriate speed strategy
    :param cx: x of reference path [m]
    :param cy: y of reference path [m]
    :param cyaw: yaw of reference path [m]
    :param target_speed: target speed [m/s]
    :return: speed profile
    """

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi

    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle
#%%
cx=path.x
cy=path.y
cyaw=path.yaw
sp = calc_speed_profile(cx, cy, cyaw, P.target_speed)

ref_path = PATH(cx, cy, cyaw)
node = Node(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

time = 0.0
x = [node.x]
y = [node.y]
yaw = [node.yaw]
v = [node.v]
t = [0.0]
d = [0.0]
a = [0.0]
#%%
delta_opt, a_opt = None, None
a_exc, delta_exc = 0.0, 0.0

while time < P.time_max:
    z_ref, target_ind = \
        calc_ref_trajectory_in_T_step(node, ref_path, sp)

    z0 = [node.x, node.y, node.v, node.yaw]

    a_opt, delta_opt, x_opt, y_opt, yaw_opt, v_opt = \
        linear_mpc_control(z_ref, z0, a_opt, delta_opt)

    if delta_opt is not None:
        delta_exc, a_exc = delta_opt[0], a_opt[0]

    node.update(a_exc, delta_exc, 1.0)
    time += P.dt

    x.append(node.x)
    y.append(node.y)
    yaw.append(node.yaw)
    v.append(node.v)
    t.append(time)
    d.append(delta_exc)
    a.append(a_exc)

    dist = math.hypot(node.x - cx[-1], node.y - cy[-1])

    if dist < P.dist_stop and \
        abs(node.v) < P.speed_stop:
        break

    dy = (node.yaw - yaw[-2]) / (node.v * P.dt)
    steer = rs.pi_2_pi(-math.atan(P.WB * dy))
    steer = rs.pi_2_pi(-math.atan(P.WB * dy))

    plt.cla()
    draw.draw_car(node.x, node.y, node.yaw, steer, P)
    plt.gcf().canvas.mpl_connect('key_release_event',
                                 lambda event:
                                 [exit(0) if event.key == 'escape' else None])

    if x_opt is not None:
        plt.plot(x_opt, y_opt, color='darkviolet', marker='*')
    plt.plot(ox, oy, "sk")
    plt.plot(cx, cy, color='gray')
    plt.plot(x, y, '-b')
    plt.plot(cx[target_ind], cy[target_ind])
    plt.axis("equal")
    plt.title("Linear MPC, " + "v = " + str(round(node.v * 3.6, 2)))
    plt.pause(0.001)

plt.show()