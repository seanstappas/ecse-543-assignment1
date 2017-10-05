from __future__ import division

import copy
import random
from abc import ABCMeta, abstractmethod
from matrices import Matrix


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


class Boundary:
    __metaclass__ = ABCMeta

    @abstractmethod
    def potential(self):
        raise NotImplementedError

    @abstractmethod
    def contains_point(self, x, y):
        raise NotImplementedError


class OuterConductorBoundary(Boundary):
    def potential(self):
        return 0

    def contains_point(self, x, y):
        return x == 0 or y == 0


class QuarterInnerConductorBoundary(Boundary):
    def potential(self):
        return 15

    def contains_point(self, x, y):
        return x >= 0.06 and y >= 0.06


class Guesser:
    __metaclass__ = ABCMeta

    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    @abstractmethod
    def guess(self, x, y):
        raise NotImplementedError


class RandomGuesser(Guesser):
    def guess(self, x, y):
        return random.randint(self.minimum, self.maximum)


class LinearGuesser(Guesser):
    def guess(self, x, y):
        return 150 * x if x < 0.06 else 150 * y


class MeshConstructor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def construct_mesh(self, h):
        raise NotImplementedError


class CoaxialCableMeshConstructor(MeshConstructor):
    def __init__(self):
        outer_boundary = OuterConductorBoundary()
        inner_boundary = QuarterInnerConductorBoundary()
        self.boundaries = (inner_boundary, outer_boundary)
        self.guesser = LinearGuesser(0, 15)
        self.boundary_size = 0.1

    def construct_mesh(self, h):
        num_mesh_points_along_axis = int(self.boundary_size / h) + 1
        phi = Matrix.empty(num_mesh_points_along_axis, num_mesh_points_along_axis)
        for i in range(num_mesh_points_along_axis):
            for j in range(num_mesh_points_along_axis):
                x = i * h
                y = j * h
                phi[i][j] = self.guesser.guess(x, y)
                for boundary in self.boundaries:
                    if boundary.contains_point(x, y):
                        phi[i][j] = boundary.potential()
        return phi


class IterativeRelaxer:
    def __init__(self, relaxer, epsilon, phi, h):
        self.relaxer = relaxer
        self.epsilon = epsilon
        self.phi = phi
        self.boundary = QuarterInnerConductorBoundary()
        self.h = h

    def relaxation(self):
        phi_new = copy.deepcopy(self.phi)
        for i in range(1, len(self.phi) - 1):
            for j in range(1, len(self.phi[0]) - 1):
                x = i * self.h
                y = j * self.h
                if not self.boundary.contains_point(x, y):
                    phi_new[i][j] = self.relaxer.relax(self.phi, phi_new, i, j)
        self.phi = phi_new
        if not self.convergence():
            self.relaxation()

    def convergence(self):
        for i in range(1, len(self.phi) - 1):
            for j in range(1, len(self.phi[0]) - 1):
                x = i * self.h
                y = j * self.h
                if not self.boundary.contains_point(x, y) and self.residual(i, j) > self.epsilon:
                    return False
        return True

    def residual(self, i, j):
        return self.phi[i + 1][j] + self.phi[i - 1][j] + self.phi[i][j + 1] + self.phi[i][j - 1] - 4 * self.phi[i][j]

