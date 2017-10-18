from __future__ import division

import csv
from matrices import Matrix
from choleski import choleski_solve


def solve_linear_network(A, Y, J, E, half_bandwidth=None):
    """
    Solve the linear resistive network described by the given matrices.

    :param A: the incidence matrix
    :param Y: the admittance matrix
    :param J: the current source matrix
    :param E: the voltage source matrix
    :param half_bandwidth:
    :return: the solved voltage matrix
    """
    A_new = A * Y * A.transpose()
    b = A * (J - Y * E)
    return choleski_solve(A_new, b, half_bandwidth=half_bandwidth)


def csv_to_network_branch_matrices(filename):
    """
    Converts a CSV file to Y, J, E network matrices.

    :param filename: the name of the CSV file
    :return: the Y, J, E network matrices
    """
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        J = []
        Y = []
        E = []
        for row in reader:
            J_k = float(row[0])
            R_k = float(row[1])
            E_k = float(row[2])
            J.append(J_k)
            Y.append(1 / R_k)
            E.append(E_k)
        Y = Matrix.diagonal(Y)
        J = Matrix.column_vector(J)
        E = Matrix.column_vector(E)
        return Y, J, E


def create_network_matrices_mesh(rows, cols, branch_resistance, test_current):
    """
    Create the network matrices needed (A, Y, J, E) to solve the resitive mesh network with the given rows, columns,
    branch resistance and test current.

    :param rows: the number of rows in the mesh
    :param cols: the number of columns in the mesh
    :param branch_resistance: the resistance in each branch
    :param test_current: the test current to apply
    :return: the network matrices (A, Y, J, E)
    """
    num_horizontal_branches = (cols - 1) * rows
    num_vertical_branches = (rows - 1) * cols
    num_branches = num_horizontal_branches + num_vertical_branches + 1
    num_nodes = rows * cols - 1

    A = create_incidence_matrix_mesh(cols, num_branches, num_horizontal_branches, num_nodes, num_vertical_branches)
    Y, J, E = create_network_branch_matrices_mesh(num_branches, branch_resistance, test_current)

    return A, Y, J, E


def create_incidence_matrix_mesh(cols, num_branches, num_horizontal_branches, num_nodes, num_vertical_branches):
    """
    Create the incidence matrix given by the resistive mesh with the given number of columns, number of branches,
    number of horizontal branches, number of nodes, and number of vertical branches.

    :param cols: the number of columns in the mesh
    :param num_branches: the number of branches in the mesh
    :param num_horizontal_branches: the number of horizontal branches in the mesh
    :param num_nodes: the number of nodes in the mesh
    :param num_vertical_branches: the number of vertical branches in the mesh
    :return: the incidence matrix (A)
    """
    A = Matrix.empty(num_nodes, num_branches)
    node_offset = -1
    for branch in range(num_horizontal_branches):
        if branch == num_horizontal_branches - cols + 1:
            A[branch + node_offset + 1][branch] = 1
        else:
            if branch % (cols - 1) == 0:
                node_offset += 1
            node_number = branch + node_offset
            A[node_number][branch] = -1
            A[node_number + 1][branch] = 1
    branch_offset = num_horizontal_branches
    node_offset = cols
    for branch in range(num_vertical_branches):
        if branch == num_vertical_branches - cols:
            node_offset -= 1
            A[branch][branch + branch_offset] = 1
        else:
            A[branch][branch + branch_offset] = 1
            A[branch + node_offset][branch + branch_offset] = -1
    if num_branches == 2:
        A[0][1] = -1
    else:
        A[cols - 1][num_branches - 1] = -1
    return A


def create_network_branch_matrices_mesh(num_branches, branch_resistance, test_current):
    """
    Create the Y, J, E network branch matrices of the resistive mesh given by the provided number of branches, branch
    resistance and test current.

    :param num_branches: the number of branches in the mesh
    :param branch_resistance: the resistance of each branch in the mesh
    :param test_current: the test current to apply to the mesh
    :return: the Y, J, E network branch matrices
    """
    Y = Matrix.diagonal([1 / branch_resistance if branch < num_branches - 1 else 0 for branch in range(num_branches)])
    # Negative test current here because we assume current is coming OUT of the test current node.
    J = Matrix.column_vector([0 if branch < num_branches - 1 else -test_current for branch in range(num_branches)])
    E = Matrix.column_vector([0 for branch in range(num_branches)])
    return Y, J, E


def find_mesh_resistance(N, branch_resistance, half_bandwidth=None):
    """
    Find the equivalent resistance of an Nx2N resistive mesh with the given branch resistance and optional
    half-bandwidth

    :param N: the size of the mesh (Nx2N)
    :param branch_resistance: the resistance of each branch of the mesh
    :param half_bandwidth: the half-bandwidth to be used for banded Choleski decomposition (or None to use non-banded)
    :return: the equivalent resistance of the mesh
    """
    test_current = 0.01
    A, Y, J, E = create_network_matrices_mesh(N, 2 * N, branch_resistance, test_current)
    x = solve_linear_network(A, Y, J, E, half_bandwidth=half_bandwidth)
    test_voltage = x[2 * N - 1 if N > 1 else 0][0]
    equivalent_resistance = test_voltage / test_current
    return equivalent_resistance
