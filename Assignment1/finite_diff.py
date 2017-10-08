from __future__ import division

import copy
import random
from abc import ABCMeta, abstractmethod

import time

import math

from matrices import Matrix


class Relaxer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def relax(self, phi, i, j):
        raise NotImplementedError


class SimpleRelaxer(Relaxer):
    """Relaxer which can represent a Jacobi relaxer, if the 'old' phi is given, or a Gauss-Seidel relaxer, if phi is
    modified in place."""
    def relax(self, phi, i, j):
        return (phi[i + 1][j] + phi[i - 1][j] + phi[i][j + 1] + phi[i][j - 1]) / 4


class SuccessiveOverRelaxer(Relaxer):
    def __init__(self, omega):
        self.gauss_seidel = SimpleRelaxer()
        self.omega = omega

    def relax(self, phi, i, j):
        return (1 - self.omega) * phi[i][j] + self.omega * self.gauss_seidel.relax(phi, i, j)


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
        return x == 0 or y == 0 or x == 0.2 or y == 0.2


class QuarterInnerConductorBoundary(Boundary):
    def potential(self):
        return 15

    def contains_point(self, x, y):
        return 0.06 <= x <= 0.14 and 0.08 <= y <= 0.12


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


def radial(k, x, y, x_source, y_source):
    return k / (math.sqrt((x_source - x)**2 + (y_source - y)**2))


class RadialGuesser(Guesser):
    def guess(self, x, y):
        return 0.0225 * (radial(20, x, y, 0.1, 0.1) - radial(1, x, y, 0, y) - radial(1, x, y, x, 0))


class CoaxialCableMeshConstructor:
    def __init__(self):
        outer_boundary = OuterConductorBoundary()
        inner_boundary = QuarterInnerConductorBoundary()
        self.boundaries = (inner_boundary, outer_boundary)
        self.guesser = RadialGuesser(0, 15)
        self.boundary_size = 0.2

    def construct_simple_mesh(self, h):
        num_mesh_points_along_axis = int(self.boundary_size / h) + 1
        phi = Matrix.empty(num_mesh_points_along_axis, num_mesh_points_along_axis)
        for i in range(num_mesh_points_along_axis):
            y = i * h
            for j in range(num_mesh_points_along_axis):
                x = j * h
                boundary_pt = False
                for boundary in self.boundaries:
                    if boundary.contains_point(x, y):
                        boundary_pt = True
                        phi[i][j] = boundary.potential()
                if not boundary_pt:
                    phi[i][j] = self.guesser.guess(x, y)
        return phi

    def construct_symmetric_mesh(self, h):
        max_index = int(0.1 / h) + 2  # Only need to store up to middle
        phi = Matrix.empty(max_index, max_index)
        for i in range(max_index):
            y = i * h
            for j in range(max_index):
                x = j * h
                boundary_pt = False
                for boundary in self.boundaries:
                    if boundary.contains_point(x, y):
                        boundary_pt = True
                        phi[i][j] = boundary.potential()
                if not boundary_pt:
                    phi[i][j] = self.guesser.guess(x, y)
        return phi


def point_to_indices(x, y, h):
    i = int(y / h)
    j = int(x / h)
    return i, j


class IterativeRelaxer:
    def __init__(self, relaxer, epsilon, phi, h):
        self.relaxer = relaxer
        self.epsilon = epsilon
        self.phi = phi
        self.boundary = QuarterInnerConductorBoundary()
        self.h = h
        self.num_iterations = 0
        self.rows = len(phi)
        self.cols = len(phi[0])
        self.mid_index = int(0.1 / h)

    def relaxation_jacobi(self):
        # t = time.time()

        while not self.convergence():
            self.num_iterations += 1

            last_row = [0] * (self.cols - 1)
            for i in range(1, self.rows - 1):
                y = i * self.h
                for j in range(1, self.cols - 1):
                    x = j * self.h
                    if not self.boundary.contains_point(x, y):
                        last_val = last_row[j - 2] if j > 1 else 0
                        relaxed_value = (self.phi[i + 1][j] + last_row[j - 1] + self.phi[i][j + 1] + last_val) / 4
                        last_row[j - 1] = self.phi[i][j]
                        self.phi[i][j] = relaxed_value
                        if i == self.mid_index - 1:
                            self.phi[i + 2][j] = relaxed_value
                        elif j == self.mid_index - 1:
                            self.phi[i][j + 2] = relaxed_value

        # print('Runtime: {} s'.format(time.time() - t))

    def relaxation_sor(self):
        while not self.convergence():
            self.num_iterations += 1
            for i in range(1, self.rows - 1):
                y = i * self.h
                for j in range(1, self.cols - 1):
                    x = j * self.h
                    if not self.boundary.contains_point(x, y):
                        relaxed_value = self.relaxer.relax(self.phi, i, j)
                        self.phi[i][j] = relaxed_value
                        if i == self.mid_index - 1:
                            self.phi[i + 2][j] = relaxed_value
                        elif j == self.mid_index - 1:
                            self.phi[i][j + 2] = relaxed_value

    def convergence(self):
        max_i, max_j = point_to_indices(0.1, 0.1, self.h)
        # Only need to compute for 1/4 of grid
        for i in range(1, max_i + 1):
            y = i * self.h
            for j in range(1, max_j + 1):
                x = j * self.h
                if not self.boundary.contains_point(x, y) and self.residual(i, j) >= self.epsilon:
                    return False
        return True

    def residual(self, i, j):
        return abs(self.phi[i+1][j] + self.phi[i-1][j] + self.phi[i][j+1] + self.phi[i][j-1] - 4 * self.phi[i][j])

    def get_potential(self, x, y):
        i, j = point_to_indices(x, y, self.h)
        return self.phi[i][j]

    def print_grid(self):
        header = ''
        for j in range(len(self.phi[0])):
            y = j * self.h
            header += '{:6.2f} '.format(y)
        print(header)
        print(self.phi)
        # for i in range(len(self.phi)):
        #     x = i * self.h
        #     print('{:6.2f} '.format(x))


def successive_over_relaxation(omega, epsilon, phi, h):
    relaxer = SuccessiveOverRelaxer(omega)
    iter_relaxer = IterativeRelaxer(relaxer, epsilon, phi, h)
    iter_relaxer.relaxation_sor()
    return iter_relaxer


def jacobi_relaxation(epsilon, phi, h):
    relaxer = SimpleRelaxer()
    iter_relaxer = IterativeRelaxer(relaxer, epsilon, phi, h)
    iter_relaxer.relaxation_jacobi()
    return iter_relaxer
