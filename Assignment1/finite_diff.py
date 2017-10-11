from __future__ import division

import math
import random
from abc import ABCMeta, abstractmethod

from matrices import Matrix


class Relaxer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def relax(self, phi, i, j):
        raise NotImplementedError

    def reset(self):
        pass

    def residual(self, phi, i, j):
        return abs(phi[i + 1][j] + phi[i - 1][j] + phi[i][j + 1] + phi[i][j - 1] - 4 * phi[i][j])


class GaussSeidelRelaxer(Relaxer):
    """Relaxer which can represent a Jacobi relaxer, if the 'old' phi is given, or a Gauss-Seidel relaxer, if phi is
    modified in place."""

    def relax(self, phi, i, j):
        return (phi[i + 1][j] + phi[i - 1][j] + phi[i][j + 1] + phi[i][j - 1]) / 4


class JacobiRelaxer(Relaxer):
    def __init__(self, num_cols):
        self.num_cols = num_cols
        self.prev_row = [0] * (num_cols - 1)  # Don't need to copy entire phi, just previous row

    def relax(self, phi, i, j):
        left_val = self.prev_row[j - 2] if j > 1 else 0
        top_val = self.prev_row[j - 1]
        self.prev_row[j - 1] = phi[i][j]
        return (phi[i + 1][j] + top_val + phi[i][j + 1] + left_val) / 4

    def reset(self):
        self.prev_row = [0] * (self.num_cols - 1)


class NonUniformRelaxer(Relaxer):
    def __init__(self, mesh):
        self.mesh = mesh

    def get_distances(self, i, j):
        a1 = self.mesh.get_y(i) - self.mesh.get_y(i - 1)
        a2 = self.mesh.get_y(i + 1) - self.mesh.get_y(i)
        b1 = self.mesh.get_x(j) - self.mesh.get_x(j - 1)
        b2 = self.mesh.get_x(j + 1) - self.mesh.get_x(j)
        return a1, a2, b1, b2

    def relax(self, phi, i, j):
        a1, a2, b1, b2 = self.get_distances(i, j)

        return ((phi[i - 1][j] / a1 + phi[i + 1][j] / a2) / (a1 + a2)
                + (phi[i][j - 1] / b1 + phi[i][j + 1] / b2) / (b1 + b2)) / (1 / (a1 * a2) + 1 / (b1 * b2))

    def residual(self, phi, i, j):
        a1, a2, b1, b2 = self.get_distances(i, j)

        return abs(((phi[i - 1][j] / a1 + phi[i + 1][j] / a2) / (a1 + a2)
                    + (phi[i][j - 1] / b1 + phi[i][j + 1] / b2) / (b1 + b2))
                   - phi[i][j] * (1 / (a1 * a2) + 1 / (b1 * b2)))


class SuccessiveOverRelaxer(Relaxer):
    def __init__(self, omega):
        self.gauss_seidel = GaussSeidelRelaxer()
        self.omega = omega

    def relax(self, phi, i, j, last_row=None, a1=None, a2=None, b1=None, b2=None):
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


class PotentialGuesser:
    __metaclass__ = ABCMeta

    def __init__(self, min_potential, max_potential):
        self.min_potential = min_potential
        self.max_potential = max_potential

    @abstractmethod
    def guess(self, x, y):
        raise NotImplementedError


class RandomPotentialGuesser(PotentialGuesser):
    def guess(self, x, y):
        return random.randint(self.min_potential, self.max_potential)


class LinearPotentialGuesser(PotentialGuesser):
    def guess(self, x, y):
        return 150 * x if x < 0.06 else 150 * y


def radial(k, x, y, x_source, y_source):
    return k / (math.sqrt((x_source - x) ** 2 + (y_source - y) ** 2))


class RadialPotentialGuesser(PotentialGuesser):
    def guess(self, x, y):
        return 0.0225 * (radial(20, x, y, 0.1, 0.1) - radial(1, x, y, 0, y) - radial(1, x, y, x, 0))


class PhiConstructor:
    def __init__(self):
        outer_boundary = OuterConductorBoundary()
        inner_boundary = QuarterInnerConductorBoundary()
        self.boundaries = (inner_boundary, outer_boundary)
        self.guesser = RadialPotentialGuesser(0, 15)
        self.boundary_size = 0.2

    def construct_simple_phi(self, h):
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

    def construct_symmetric_phi(self, h):
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


class Mesh:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_x(self, j):
        raise NotImplementedError

    @abstractmethod
    def get_y(self, i):
        raise NotImplementedError

    @abstractmethod
    def get_i(self, y):
        raise NotImplementedError

    @abstractmethod
    def get_j(self, x):
        raise NotImplementedError

    def point_to_indices(self, x, y):
        return self.get_i(y), self.get_j(x)

    def indices_to_points(self, i, j):
        return self.get_x(j), self.get_y(i)


class SimpleMesh(Mesh):
    def __init__(self, h):
        self.h = h

    def get_i(self, y):
        return int(y / self.h)

    def get_j(self, x):
        return int(x / self.h)

    def get_x(self, j):
        return j * self.h

    def get_y(self, i):
        return i * self.h


class NonUniformMesh(Mesh):
    def __init__(self, x_values, y_values):
        self.x_values = x_values
        self.y_values = y_values

    def get_i(self, y):
        return self.y_values.index(y)

    def get_j(self, x):
        return self.x_values.index(x)

    def get_x(self, j):
        return self.x_values[j]

    def get_y(self, i):
        return self.y_values[i]


class IterativeRelaxer:
    def __init__(self, relaxer, epsilon, phi, h, mesh):
        self.relaxer = relaxer
        self.epsilon = epsilon
        self.phi = phi
        self.boundary = QuarterInnerConductorBoundary()
        self.h = h
        self.num_iterations = 0
        self.rows = len(phi)
        self.cols = len(phi[0])
        self.mid_index = int(0.1 / h)
        self.mesh = mesh

    def relaxation(self):
        while not self.convergence():
            self.num_iterations += 1
            for i in range(1, self.rows - 1):
                y = self.mesh.get_y(i)
                for j in range(1, self.cols - 1):
                    x = self.mesh.get_x(j)
                    if not self.boundary.contains_point(x, y):
                        relaxed_value = self.relaxer.relax(self.phi, i, j)
                        self.phi[i][j] = relaxed_value
                        if i == self.mid_index - 1:
                            self.phi[i + 2][j] = relaxed_value
                        elif j == self.mid_index - 1:
                            self.phi[i][j + 2] = relaxed_value
            self.relaxer.reset()

    def convergence(self):
        max_i, max_j = self.mesh.point_to_indices(0.1, 0.1)
        # Only need to compute for 1/4 of grid
        for i in range(1, max_i + 1):
            y = self.mesh.get_y(i)
            for j in range(1, max_j + 1):
                x = self.mesh.get_x(j)
                if not self.boundary.contains_point(x, y) and self.relaxer.residual(self.phi, i, j) >= self.epsilon:
                    return False
        return True

    def get_potential(self, x, y):
        i, j = self.mesh.point_to_indices(x, y)
        return self.phi[i][j]


def non_uniform_successive_over_relaxation(epsilon, x_values, y_values):
    h = 0.1 / (len(x_values) - 2)  # As if h uniform, but actually x, y values clustered around (0.06, 0.04)
    phi = PhiConstructor().construct_symmetric_phi(h)
    mesh = NonUniformMesh(x_values, y_values)
    relaxer = NonUniformRelaxer(mesh)
    iter_relaxer = IterativeRelaxer(relaxer, epsilon, phi, h, mesh)
    iter_relaxer.relaxation()
    return iter_relaxer


def successive_over_relaxation(omega, epsilon, h):
    phi = PhiConstructor().construct_symmetric_phi(h)
    relaxer = SuccessiveOverRelaxer(omega)
    mesh = SimpleMesh(h)
    iter_relaxer = IterativeRelaxer(relaxer, epsilon, phi, h, mesh)
    iter_relaxer.relaxation()
    return iter_relaxer


def jacobi_relaxation(epsilon, h):
    phi = PhiConstructor().construct_symmetric_phi(h)
    relaxer = GaussSeidelRelaxer()
    mesh = SimpleMesh(h)
    iter_relaxer = IterativeRelaxer(relaxer, epsilon, phi, h, mesh)
    iter_relaxer.relaxation()
    return iter_relaxer
