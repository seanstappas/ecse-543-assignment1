from __future__ import division

import math
import random
from abc import ABCMeta, abstractmethod

from matrices import Matrix

MESH_SIZE = 0.2


class Relaxer:
    """
    Performs the relaxing stage of the finite difference method.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def relax(self, phi, i, j):
        """
        Perform a relaxation iteration on a given (i, j) point of the given phi matrix.

        :param phi: the phi matrix
        :param i: the row index
        :param j: the column index
        """
        raise NotImplementedError

    def reset(self):
        """
        Optional method to reset the relaxer.
        """
        pass

    def residual(self, phi, i, j):
        """
        Calculate the residual at the given (i, j) point of the given phi matrix.

        :param phi: the phi matrix
        :param i: the row index
        :param j: the column index
        :return:
        """
        return abs(phi[i + 1][j] + phi[i - 1][j] + phi[i][j + 1] + phi[i][j - 1] - 4 * phi[i][j])


class GaussSeidelRelaxer(Relaxer):
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
    """
    Constant-potential boundary in the finite difference mesh, representing a conductor.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def potential(self):
        """
        Return the potential on the boundary.
        """
        raise NotImplementedError

    @abstractmethod
    def contains_point(self, x, y):
        """
        Returns true if the boundary contains the given (x, y) point.

        :param x: the x coordinate of the point
        :param y: the y coordinate of the point
        """
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
    """
    Guesses the initial potential in the finite-difference mesh.
    """
    __metaclass__ = ABCMeta

    def __init__(self, min_potential, max_potential):
        self.min_potential = min_potential
        self.max_potential = max_potential

    @abstractmethod
    def guess(self, x, y):
        """
        Guess the potential at the given (x, y) point, and return it.

        :param x: the x coordinate of the point
        :param y: the y coordinate of the point
        """
        raise NotImplementedError


class RandomPotentialGuesser(PotentialGuesser):
    def guess(self, x, y):
        return random.randint(self.min_potential, self.max_potential)


class LinearPotentialGuesser(PotentialGuesser):
    def guess(self, x, y):
        return 150 * x if x < 0.06 else 150 * y


class RadialPotentialGuesser(PotentialGuesser):
    def guess(self, x, y):
        def radial(k, x, y, x_source, y_source):
            return k / (math.sqrt((x_source - x) ** 2 + (y_source - y) ** 2))

        return 0.0225 * (radial(20, x, y, 0.1, 0.1) - radial(1, x, y, 0, y) - radial(1, x, y, x, 0))


class PhiConstructor:
    """
    Constructs the phi potential matrix with an outer conductor, inner conductor, mesh points and an inital potential
    guess.
    """

    def __init__(self, mesh):
        outer_boundary = OuterConductorBoundary()
        inner_boundary = QuarterInnerConductorBoundary()
        self.boundaries = (inner_boundary, outer_boundary)
        self.guesser = RadialPotentialGuesser(0, 15)
        self.mesh = mesh

    def construct_phi(self):
        phi = Matrix.empty(self.mesh.num_rows, self.mesh.num_cols)
        for i in range(self.mesh.num_rows):
            y = self.mesh.get_y(i)
            for j in range(self.mesh.num_cols):
                x = self.mesh.get_x(j)
                boundary_pt = False
                for boundary in self.boundaries:
                    if boundary.contains_point(x, y):
                        boundary_pt = True
                        phi[i][j] = boundary.potential()
                if not boundary_pt:
                    phi[i][j] = self.guesser.guess(x, y)
        return phi


class SquareMeshConstructor:
    """
    Constructs a square mesh.
    """

    def __init__(self, size):
        self.size = size

    def construct_uniform_mesh(self, h):
        """
        Constructs a uniform mesh with the given node spacing.

        :param h: the node spacing
        :return: the constructed mesh
        """
        num_rows = num_cols = int(self.size / h) + 1
        return SimpleMesh(h, num_rows, num_cols)

    def construct_symmetric_uniform_mesh(self, h):
        """
        Construct a symmetric uniform mesh with the given node spacing.

        :param h: the node spacing
        :return: the constructed mesh
        """
        half_size = self.size / 2
        num_rows = num_cols = int(half_size / h) + 2  # Only need to store up to middle
        return SimpleMesh(h, num_rows, num_cols)

    def construct_symmetric_non_uniform_mesh(self, x_values, y_values):
        """
        Construct a symmetric non-uniform mesh with the given adjacent x coordinates and y coordinates.

        :param x_values: the values of successive x coordinates
        :param y_values: the values of successive y coordinates
        :return: the constructed mesh
        """
        return NonUniformMesh(x_values, y_values)


class Mesh:
    """
    Finite-difference mesh.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_x(self, j):
        """
        Get the x value at the specified index.

        :param j: the column index.
        """
        raise NotImplementedError

    @abstractmethod
    def get_y(self, i):
        """
        Get the y value at the specified index.

        :param i: the row index.
        """
        raise NotImplementedError

    @abstractmethod
    def get_i(self, y):
        """
        Get the row index of the specified y coordinate.

        :param y: the y coordinate
        """
        raise NotImplementedError

    @abstractmethod
    def get_j(self, x):
        """
        Get the column index of the specified x coordinate.

        :param x: the x coordinate
        """
        raise NotImplementedError

    def point_to_indices(self, x, y):
        """
        Converts the given (x, y) point to (i, j) matrix indices.

        :param x: the x coordinate
        :param y: the y coordinate
        :return: the (i, j) matrix indices
        """
        return self.get_i(y), self.get_j(x)

    def indices_to_points(self, i, j):
        """
        Converts the given (i, j) matrix indices to an (x, y) point.

        :param i: the row index
        :param j: the column index
        :return: the (x, y) point
        """
        return self.get_x(j), self.get_y(i)


class SimpleMesh(Mesh):
    def __init__(self, h, num_rows, num_cols):
        self.h = h
        self.num_rows = num_rows
        self.num_cols = num_cols

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
        self.num_rows = len(y_values)
        self.num_cols = len(x_values)

    def get_i(self, y):
        return self.y_values.index(y)

    def get_j(self, x):
        return self.x_values.index(x)

    def get_x(self, j):
        return self.x_values[j]

    def get_y(self, i):
        return self.y_values[i]


class IterativeRelaxer:
    """
    Performs finite-difference iterative relaxation on a phi potential matrix associated with a mesh.
    """

    def __init__(self, relaxer, epsilon, phi, mesh):
        self.relaxer = relaxer
        self.epsilon = epsilon
        self.phi = phi
        self.boundary = QuarterInnerConductorBoundary()
        self.num_iterations = 0
        self.rows = len(phi)
        self.cols = len(phi[0])
        self.mesh = mesh
        self.mid_i = mesh.get_i(MESH_SIZE / 2)
        self.mid_j = mesh.get_j(MESH_SIZE / 2)

    def relaxation(self):
        """
        Performs iterative relaxation until convergence is met.

        :return: the current iterative relaxer object
        """
        while not self.convergence():
            self.num_iterations += 1
            self.relaxation_iteration()
            self.relaxer.reset()
        return self

    def relaxation_iteration(self):
        """
        Performs one iteration of relaxation.
        """
        for i in range(1, self.rows - 1):
            y = self.mesh.get_y(i)
            for j in range(1, self.cols - 1):
                x = self.mesh.get_x(j)
                if not self.boundary.contains_point(x, y):
                    relaxed_value = self.relaxer.relax(self.phi, i, j)
                    self.phi[i][j] = relaxed_value
                    if i == self.mid_i - 1:
                        self.phi[i + 2][j] = relaxed_value
                    elif j == self.mid_j - 1:
                        self.phi[i][j + 2] = relaxed_value

    def convergence(self):
        """
        Checks if the phi matrix has reached convergence.

        :return: True if the phi matrix has reached convergence, False otherwise
        """
        max_i, max_j = self.mesh.point_to_indices(0.1, 0.1)  # Only need to compute for 1/4 of grid
        for i in range(1, max_i + 1):
            y = self.mesh.get_y(i)
            for j in range(1, max_j + 1):
                x = self.mesh.get_x(j)
                if not self.boundary.contains_point(x, y) and self.relaxer.residual(self.phi, i, j) >= self.epsilon:
                    return False
        return True

    def get_potential(self, x, y):
        """
        Get the potential at the given (x, y) point.

        :param x: the x coordinate
        :param y: the y coordinate
        :return: the potential at the given (x, y) point
        """
        i, j = self.mesh.point_to_indices(x, y)
        return self.phi[i][j]


def non_uniform_jacobi(epsilon, x_values, y_values):
    """
    Perform Jacobi relaxation on a non-uniform finite-difference mesh.

    :param epsilon: the maximum error to achieve convergence
    :param x_values: the values of successive x coordinates
    :param y_values: the values of successive y coordinates
    :return: the relaxer object
    """
    mesh = SquareMeshConstructor(MESH_SIZE).construct_symmetric_non_uniform_mesh(x_values, y_values)
    relaxer = NonUniformRelaxer(mesh)
    phi = PhiConstructor(mesh).construct_phi()
    return IterativeRelaxer(relaxer, epsilon, phi, mesh).relaxation()


def successive_over_relaxation(omega, epsilon, h):
    """
    Perform SOR on a uniform symmetric finite-difference mesh.

    :param omega: the omega value for SOR
    :param epsilon: the maximum error to achieve convergence
    :param h: the node spacing
    :return: the relaxer object
    """
    mesh = SquareMeshConstructor(MESH_SIZE).construct_symmetric_uniform_mesh(h)
    relaxer = SuccessiveOverRelaxer(omega)
    phi = PhiConstructor(mesh).construct_phi()
    return IterativeRelaxer(relaxer, epsilon, phi, mesh).relaxation()


def jacobi_relaxation(epsilon, h):
    """
    Perform Jacobi relaxation on a uniform symmetric finite-difference mesh.

    :param epsilon: the maximum error to achieve convergence
    :param h: the node spacing
    :return: the relaxer object
    """
    mesh = SquareMeshConstructor(MESH_SIZE).construct_symmetric_uniform_mesh(h)
    relaxer = GaussSeidelRelaxer()
    phi = PhiConstructor(mesh).construct_phi()
    return IterativeRelaxer(relaxer, epsilon, phi, mesh).relaxation()
