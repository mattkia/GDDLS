import torch

import numpy as np

from sklearn.metrics import euclidean_distances


def make_dx(size):
    dx = torch.zeros([size, size])
    dx[0, 0] = -1
    dx[0, 1] = 1
    dx[-1, -1] = 1
    dx[-1, -2] = -1

    for i in range(1, size - 1):
        dx[i, i - 1] = -0.5
        dx[i, i + 1] = 0.5

    return dx


def make_dy(size):
    dy = torch.zeros([size, size])
    dy[0, 0] = -1
    dy[1, 0] = 1
    dy[-2, -1] = -1
    dy[-1, -1] = 1

    for j in range(1, size - 1):
        dy[j - 1, j] = -0.5
        dy[j + 1, j] = 0.5

    return dy


def make_dxx(size):
    dxx = torch.zeros([size, size])
    dxx[0, 0] = -2
    dxx[0, 1] = 1
    dxx[-1, -1] = -2
    dxx[-1, -2] = 1

    for i in range(1, size - 1):
        dxx[i, i - 1] = 1
        dxx[i, i] = -2
        dxx[i, i + 1] = 1

    return dxx


def make_dxy(size):
    dxy = torch.zeros([size, size])
    dxy[0, 0] = -1
    dxy[0, 1] = 1
    dxy[-1, -1] = 1
    dxy[-1, -2] = -1

    for i in range(1, size - 1):
        dxy[i, i - 1] = -0.5
        dxy[i, i + 1] = 0.5

    dyx = torch.zeros([size, size])
    dyx[0, 0] = -1
    dyx[1, 0] = 1
    dyx[-2, -1] = -1
    dyx[-1, -1] = 1

    for j in range(1, size - 1):
        dyx[j - 1, j] = -0.5
        dyx[j + 1, j] = 0.5

    return dxy, dyx


def make_dyy(size):
    dyy = torch.zeros([size, size])
    dyy[0, 0] = -2
    dyy[0, 1] = 1
    dyy[-1, -1] = -2
    dyy[-2, -1] = 1

    for j in range(1, size - 1):
        dyy[j - 1, j] = 1
        dyy[j, j] = -2
        dyy[j + 1, j] = 1

    return dyy


def derivative_x(surface, dx):
    return torch.matmul(dx, surface)


def derivative_y(surface, dy):
    return torch.matmul(surface, dy)


def derivative_xx(surface, dxx):
    return torch.matmul(dxx, surface)


def derivative_xy(surface, dxy, dyx):
    out = torch.matmul(dxy, surface)

    return torch.matmul(out, dyx)


def derivative_yy(surface, dyy):
    return torch.matmul(surface, dyy)


def first_fundamental_e(surface, dx):
    phi_x = derivative_x(surface[0, 0], dx)

    return 1. + phi_x ** 2


def first_fundamental_f(surface, dx, dy):
    phi_x = derivative_x(surface[0, 0], dx)
    phi_y = derivative_y(surface[0, 0], dy)

    return phi_x * phi_y


def first_fundamental_g(surface, dy):
    phi_y = derivative_y(surface[0, 0], dy)

    return 1. + phi_y ** 2


def calculate_volume(surface, dx, dy):
    e = first_fundamental_e(surface, dx)
    f = first_fundamental_f(surface, dx, dy)
    g = first_fundamental_g(surface, dy)

    return e * g - f ** 2


def gaussian_curvature(surface, dx, dy, dxx, dxy, dyx, dyy):
    volume = calculate_volume(surface, dx, dy)
    numerator = derivative_xx(surface[0, 0], dxx) * derivative_yy(surface[0, 0], dyy) - \
                derivative_xy(surface[0, 0], dxy, dyx) ** 2

    return numerator / volume


def level_set_loss_v2(surface, ground_truth):
    epsilon = 1 / 32

    heaviside = 0.5 * (1 + (2 / np.pi) * torch.atan(surface / epsilon))
    heaviside = (heaviside - torch.min(heaviside)) / (torch.max(heaviside) - torch.min(heaviside))

    cost = torch.mean((heaviside - ground_truth) ** 2)

    return cost


def normal_vector_loss(surface, dx, dy):
    surface_x = derivative_x(surface[0, 0], dx)
    surface_y = derivative_y(surface[0, 0], dy)

    epsilon = 1 / 32

    delta = (1 / np.pi) * (epsilon / (epsilon ** 2 + surface[0, 0] ** 2))
    delta = (delta - torch.min(delta)) / (torch.max(delta) - torch.min(delta))

    phi_x = delta * surface_x
    phi_y = delta * surface_y

    cost1 = -(torch.mean(phi_x) ** 2)
    cost2 = -(torch.mean(phi_y) ** 2)

    cost = cost1 + cost2

    return cost


def curvature_loss(surface, dx, dy, dxx, dxy, dyx, dyy):
    curvature = gaussian_curvature(surface, dx, dy, dxx, dxy, dyx, dyy)

    epsilon = 1 / 32

    heaviside = 0.5 * (1 + (2 / np.pi) * torch.atan(surface[0, 0] / epsilon))
    heaviside = (heaviside - torch.min(heaviside)) / (torch.max(heaviside) - torch.min(heaviside))

    c1 = torch.mean(heaviside * curvature)
    c2 = torch.mean((1 - heaviside) * curvature)

    cost1 = torch.mean((heaviside * curvature - c1) ** 2)
    cost2 = torch.mean(((1 - heaviside) * curvature - c2) ** 2)

    cost = cost1 + cost2

    return cost


def calculate_precision_recall(heaviside, label):
    overlap = heaviside[np.where(label > 0)] > 0.1

    precision_denominator = len(heaviside[np.where(heaviside > 0.1)])
    recall_denominator = len(label[np.where(label > 0)])

    if precision_denominator != 0:
        precision = overlap.sum() / precision_denominator
    else:
        precision = 0.

    recall = overlap.sum() / recall_denominator

    return precision, recall


def calculate_intersection_over_union(heaviside, label):
    intersection = heaviside[np.where(label > 0)] > 0

    ones_matrix = np.ones(heaviside.shape)

    union = ones_matrix[(heaviside > 0.1) | (label > 0)]

    iou = intersection.sum() / union.sum()

    return iou


def calculate_omega_f_score(heaviside, label):
    d = heaviside.flatten().reshape(1, -1)
    g = label.flatten().reshape(1, -1)

    e = np.abs(g - d)

    a_matrix = np.zeros((d.shape[1], d.shape[1]))
    b_matrix = np.ones((d.shape[1], 1))

    for i in range(d.shape[1]):
        for j in range(d.shape[1]):
            loc1 = list(divmod(i, heaviside.shape[1]))
            loc2 = list(divmod(j, heaviside.shape[1]))

            if g[i, 0] == 1 and g[j, 0] == 1:
                distance = euclidean_distances([loc1], [loc2])[0][0]
                coefficient = 1 / np.sqrt(2 * np.pi * 5)
                value = coefficient * np.exp(-(distance**2 / 10))
                a_matrix[i, j] = value
            elif g[i, 0] == 0 and i == j:
                a_matrix[i, j] = 1

    f = np.minimum(e, np.matmul(e, a_matrix))
    f = np.matmul(f, b_matrix)

    tp = np.matmul(1 - f, g.T)
    fp = np.matmul(f, 1 - g.T)
    fn = np.matmul(f, g.T)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    score = (1.3 * precision * recall) / (0.3 * precision + recall)

    return score


