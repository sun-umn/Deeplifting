"""
Sensor Network Localization:


"""

# third party
import numpy as np
import torch

n = 100  # number of sensors
m = 4  # number of anchors
r = 1.0  # radio for generating the distances
p = 0.5  # probability for generating the distances


def radiogen(n, radiorange):
    """
    Function that generate the Type II sensors' positions

    """

    sensors = np.random.uniform(-0.5, 0.5, [2, n])
    anchors = np.array([[-0.45, -0.45, 0.45, 0.45], [0.45, -0.45, 0.45, -0.45]])

    m = 4

    distmat = np.zeros([n + m, n + m])

    for i in range(m):
        for j in range(n):
            dist = np.linalg.norm(anchors[:, i] - sensors[:, j])
            if dist < radiorange:
                distmat[j, n + i] = dist
            else:
                distmat[j, n + i] = 0

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(sensors[:, i] - sensors[:, j])
            if dist < radiorange:
                distmat[i, j] = dist
            else:
                distmat[i, j] = 0

    return sensors, anchors, distmat


def radiogen2(n, radiorange, p):
    """
    Function that generate the Type III sensors' positions

    """

    sensors = np.random.uniform(-0.5, 0.5, [2, n])
    anchors = np.array([[-0.45, -0.45, 0.45, 0.45], [0.45, -0.45, 0.45, -0.45]])

    m = 4

    distmat = np.zeros([n + m, n + m])

    for i in range(m):
        for j in range(n):
            dist = np.linalg.norm(anchors[:, i] - sensors[:, j])
            if dist < radiorange and np.random.uniform() < 1 - p:
                distmat[j, n + i] = dist
            else:
                distmat[j, n + i] = 0

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(sensors[:, i] - sensors[:, j])
            if dist < radiorange and np.random.uniform() < 1 - p:
                distmat[i, j] = dist
            else:
                distmat[i, j] = 0

    return sensors, anchors, distmat


radiorange = r * np.sqrt(np.log(n) / n)


sensors, anchors, distmat = radiogen(
    n, radiorange
)  # generate the sensors' positions and the distance matrix

sensors_ten = torch.tensor(sensors)
anchors_ten = torch.tensor(anchors)

I, J = distmat.nonzero()

numobs = I.shape[0]


# transform the distance matrix into a vector
distvec = np.zeros(numobs)
for numob in range(numobs):
    i = I[numob]
    j = J[numob]
    distvec[numob] = distmat[i, j]

distvec_ten = torch.tensor(distvec)


def snl_poly4(x, version='numpy'):
    """
    Function
    """

    if version == 'numpy':
        fullvecs = np.concatenate((x, anchors), axis=1)
        distvecx = np.sum(np.abs(fullvecs[:, I] - fullvecs[:, J]) ** 2, axis=0)
        result = np.linalg.norm(distvecx - distvec**2) ** 2

    elif version == 'pytorch':
        fullvecs_ten = torch.cat((x, anchors_ten), -1)
        distvecx_ten = torch.sum(
            torch.abs(fullvecs_ten[:, I] - fullvecs_ten[:, J]) ** 2, axis=0
        )
        result = torch.norm(distvecx_ten - distvec_ten**2) ** 2
    return result


# test the loss function
print(snl_poly4(sensors), snl_poly4(sensors_ten, version='pytorch'))

# generate random sensors' positions and test the function
sensors_rand = np.random.randn(2, n)
print(snl_poly4(sensors_rand))
