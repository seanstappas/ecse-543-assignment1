from __future__ import division

from linear_networks import solve_linear_network
from choleski import choleski_solve
from matrix_utilities import Matrix

L_2 = Matrix([
    [5, 0],
    [1, 3]
])
L_3 = Matrix([
    [3, 0, 0],
    [1, 2, 0],
    [8, 5, 1]
])
L_4 = Matrix([
    [1, 0, 0, 0],
    [2, 8, 0, 0],
    [5, 5, 4, 0],
    [7, 2, 8, 7]
])

matrix_2 = L_2 * L_2.transpose()
matrix_3 = L_3 * L_3.transpose()
matrix_4 = L_4 * L_4.transpose()
matrices = [matrix_2, matrix_3, matrix_4]

x_2 = Matrix.column_vector([8, 3])
x_3 = Matrix.column_vector([9, 4, 3])
x_4 = Matrix.column_vector([5, 4, 1, 9])
xs = [x_2, x_3, x_4]


def q1b():
    for count, A in enumerate(matrices):
        n = count + 2
        print('n={} matrix is positive-definite: {}'.format(n, A.is_positive_definite()))


def q1c():
    for x, A in zip(xs, matrices):
        b = A * x
        print('A: {}'.format(A))
        print('b: {}'.format(b))

        x_result = choleski_solve(A, b)
        print('x: {}'.format(x))
        print('x_result: {}'.format(x_result))  # TODO: Assert equal here (to number of sig figs)


def q1d():
    for i in range(1, 6):
        A = Matrix.csv_to_matrix('incidence_matrix_{}.csv'.format(i))
        Y, J, E = Matrix.csv_to_network_branch_matrices('network_branches_{}.csv'.format(i))
        print('Y: {}'.format(Y))
        print('J: {}'.format(J))
        print('E: {}'.format(E))
        x = solve_linear_network(A, Y, J, E)
        print('Solved x in network {}: {}'.format(i, x))  # TODO: Create my own test circuits here


if __name__ == '__main__':
    q1b()
    q1c()
    q1d()
