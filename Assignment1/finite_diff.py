import copy
from abc import ABCMeta, abstractmethod


class Relaxer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def relax(self, phi, phi_new, i, j):
        raise NotImplementedError


class JacobiRelaxer(Relaxer):
    def relax(self, phi, phi_new, i, j):
        return (phi[i + 1][j] + phi[i - 1][j] + phi[i][j + 1] + phi[i][j - 1]) / 4


class GaussSeidelRelaxer(Relaxer):
    def relax(self, phi, phi_new, i, j):
        return (phi[i + 1][j] + phi_new[i - 1][j] + phi[i][j + 1] + phi_new[i][j - 1]) / 4


class SuccessiveOverRelaxer(Relaxer):
    def __init__(self, omega):
        self.gauss_seidel = GaussSeidelRelaxer()
        self.omega = omega

    def relax(self, phi, phi_new, i, j):
        return (1 - self.omega) * phi[i][j] + self.omega * self.gauss_seidel.relax(phi, phi_new, i, j)


def relaxation(phi, epsilon, relaxer):
    phi_new = copy.deepcopy(phi)
    for i in range(1, len(phi) - 1):
        for j in range(1, len(phi[0]) - 1):
            phi_new[i][j] = relaxer.relax(phi, phi_new, i, j)
    if convergence(phi_new, epsilon):
        return phi_new
    else:
        return relaxation(phi_new, epsilon, relaxer)


def convergence(phi, epsilon):
    for i in range(1, len(phi) - 1):
        for j in range(1, len(phi[0]) - 1):
            if residual(phi, i, j) > epsilon:
                return False
    return True


def residual(phi, i, j):
    return phi[i + 1][j] + phi[i - 1][j] + phi[i][j + 1] + phi[i][j - 1] - 4 * phi[i][j]


def generate_mesh(size, spacing):
    pass
