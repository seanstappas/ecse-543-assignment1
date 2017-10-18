from __future__ import division

from csv_saver import save_rows_to_csv
from linear_networks import solve_linear_network, csv_to_network_branch_matrices
from choleski import choleski_solve
from matrices import Matrix

NETWORK_DIRECTORY = 'network_data'

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
positive_definite_matrices = [matrix_2, matrix_3, matrix_4]

x_2 = Matrix.column_vector([8, 3])
x_3 = Matrix.column_vector([9, 4, 3])
x_4 = Matrix.column_vector([5, 4, 1, 9])
xs = [x_2, x_3, x_4]


def q1():
    """
    Question 1
    """
    q1b()
    q1c()
    q1d()


def q1b():
    """
    Question 1(b): Construct some small matrices (n = 2, 3, 4, or 5) to test the program. Remember that the matrices
    must be real, symmetric and positive-definite.
    """
    print('\n=== Question 1(b) ===')
    for count, A in enumerate(positive_definite_matrices):
        n = count + 2
        print('n={} matrix is positive-definite: {}'.format(n, A.is_positive_definite()))


def q1c():
    """
    Question 1(c): Test the program you wrote in (a) with each small matrix you built in (b) in the following way:
    invent an x, multiply it by A to get b, then give A and b to your program and check that it returns x correctly.
    """
    print('\n=== Question 1(c) ===')
    n = 2
    for x, A in zip(xs, positive_definite_matrices):
        b = A * x
        print('Matrix with n={}:'.format(n))
        print('A: {}'.format(A))
        print('b: {}'.format(b))

        x_choleski = choleski_solve(A, b)
        print('Expected x: {}'.format(x))
        print('Actual x: {}'.format(x_choleski))
        n += 1


def q1d():
    """
    Question 1(d): Write a program that reads from a file a list of network branches (Jk, Rk, Ek) and a reduced
    incidence matrix, and finds the voltages at the nodes of the network. Use the code from part (a) to solve the
    matrix problem.
    """
    print('\n=== Question 1(d) ===')
    for i in range(1, 7):
        A = Matrix.csv_to_matrix('{}/incidence_matrix_{}.csv'.format(NETWORK_DIRECTORY, i))
        Y, J, E = csv_to_network_branch_matrices('{}/network_branches_{}.csv'.format(NETWORK_DIRECTORY, i))
        # print('Y: {}'.format(Y))
        # print('J: {}'.format(J))
        # print('E: {}'.format(E))
        x = solve_linear_network(A, Y, J, E)
        print('Solved for x in network {}:'.format(i))  # TODO: Create my own test circuits here
        node_numbers = []
        voltage_values = []
        for j in range(len(x)):
            print('V{} = {:.3f} V'.format(j + 1, x[j][0]))
            node_numbers.append(j + 1)
            voltage_values.append('{:.3f}'.format(x[j][0]))
        save_rows_to_csv('report/csv/q1_circuit_{}.csv'.format(i), zip(node_numbers, voltage_values),
                         header=('Node', 'Voltage (V)'))


if __name__ == '__main__':
    q1()
